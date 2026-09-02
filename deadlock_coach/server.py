"""``deadlock-coach-server`` — drop a ``.dem`` in from any laptop on the WiFi.

The browser uploads a replay, this process runs the same pipeline the CLI runs,
and then pipes the resulting report into a local coding-agent CLI for the
written review. Everything stays on this machine: the demo, the report, and the
coaching call.

Two CLIs are supported — ``claude`` and ``codex`` — and the page picks between
them per job and per question. They are interchangeable here because both are
already logged in on this machine, so switching needs no API key; what it is
actually for is running out of quota on one side mid-evening and carrying on
with the other.

Both heavy steps run as subprocesses, one job at a time:

  * parsing a 550MB demo peaks at several GB of RAM, so a crashed or leaky parse
    must not take the server down with it, and two concurrent parses would swap;
  * the model CLI is a separate binary anyway.

A single worker thread drains a FIFO queue, so an upload always returns
immediately even while an earlier replay is still parsing.
"""

# Deliberately no ``from __future__ import annotations`` here, unlike the rest
# of the package: FastAPI resolves route annotations at runtime, and stringified
# ones cannot see ``UploadFile``/``File`` because those are imported inside
# ``create_app`` to keep the module importable without the ``server`` extra.

import json
import logging
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger("deadlock-coach-server")

WEB_DIR = Path(__file__).parent / "web"
PROMPT_PATHS = {
    "en": Path(__file__).parent / "coach_prompt_en.md",
    "zh-TW": Path(__file__).parent / "coach_prompt.md",
}
CHAT_PROMPT_PATHS = {
    "en": Path(__file__).parent / "chat_prompt_en.md",
    "zh-TW": Path(__file__).parent / "chat_prompt.md",
}
LANGUAGES = frozenset(PROMPT_PATHS)
DEFAULT_LANGUAGE = "en"

#: Demos are big and slow; these bound how long a stuck job can hold the queue.
PARSE_TIMEOUT_SECONDS = 30 * 60
COACH_TIMEOUT_SECONDS = 20 * 60
CHAT_TIMEOUT_SECONDS = 10 * 60

#: Kept only for redisplay — the real conversation lives in the CLI's own
#: session, so trimming this never costs the model any context.
MAX_CHAT_TURNS = 100

#: Which CLI writes the review unless the page says otherwise.
DEFAULT_BACKEND = "claude"

#: How much of the existing conversation is handed to an engine that was not
#: there for it. Enough to keep "那第二點呢" meaningful after a switch, not so
#: much that the handover costs more than the answer.
HANDOVER_TURNS = 6
HANDOVER_CHARS = 1500

UPLOAD_CHUNK = 4 * 1024 * 1024
# Normal replays are hundreds of MB. This leaves generous headroom while
# preventing one LAN client from filling the host disk with an unbounded body.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

#: Reports run ~150KB. Anything far past that is a sign the pipeline changed
#: shape, and shipping it to the model blind would just burn tokens.
MAX_REPORT_CHARS = 400_000
MAX_VISUAL_CONTEXT_CHARS = 60_000

# A player-focused review does not benefit from 11 other players' complete
# opportunity tables or hundreds of unrelated kill-feed lines.  Global match
# sections stay intact; this only bounds the raw timeline after the selected
# player's events and public objectives have been retained.
FOCUSED_MAX_TIMELINE_EVENTS = 160
_PUBLIC_EVENT_TERMS = (
    "walker",
    "guardian",
    "barracks",
    "shrine",
    "patron",
    "mid boss",
    "urn",
    "rift",
)

#: ``\w`` is unicode-aware, so a Chinese filename survives while separators,
#: spaces and shell metacharacters do not.
_SAFE_STEM = re.compile(r"[^\w.-]+")

STATUSES = ("queued", "parsing", "awaiting_player", "coaching", "done", "failed")

_HERO_NAMES = None


def _now() -> str:
    """Local time with an offset, so the browser shows the clock the user has."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_stem(filename: str) -> str:
    """Reduce an uploaded filename to a bare, path-traversal-free stem.

    The stem ends up in a filesystem path and in the report filenames, so it has
    to survive a client sending ``../../etc/passwd`` or an empty name.
    """
    stem = Path(filename or "").name
    if stem.lower().endswith(".dem"):
        stem = stem[: -len(".dem")]
    stem = _SAFE_STEM.sub("_", stem).strip("._-")
    return stem[:80] or "replay"


@dataclass
class Job:
    """One uploaded replay and everything we produced from it."""

    id: str
    stem: str
    original_name: str
    created: str
    status: str = "queued"
    player: str | None = None
    #: The engine this job is currently on. Chosen at upload, changeable at the
    #: hero pick and again on every question.
    backend: str = DEFAULT_BACKEND
    #: Language of the generated coaching report and default chat replies.
    language: str = DEFAULT_LANGUAGE
    #: The question the user wants the first review to prioritize.
    analysis_request: str = ""
    coach: bool = True
    keep_demo: bool = False
    size_bytes: int = 0
    error: str | None = None
    log: list[str] = field(default_factory=list)
    started: str | None = None
    finished: str | None = None
    parse_seconds: float | None = None
    coach_seconds: float | None = None
    files: dict[str, str] = field(default_factory=dict)
    #: Everyone in the match, filled in after decoding so the user can point at
    #: themselves instead of the model guessing who they were.
    roster: list[dict] = field(default_factory=list)
    #: ``backend name -> session id``. Follow-up questions resume the session,
    #: so an engine never has to be shown the report twice — and each engine
    #: keeps its own, because a claude session id means nothing to codex.
    sessions: dict[str, str] = field(default_factory=dict)
    chat_turns: int = 0

    def note(self, line: str) -> None:
        self.log.append(line)
        del self.log[:-200]


def _migrated(data: dict) -> dict:
    """Bring a ``job.json`` written by an older build up to the current shape.

    Jobs on disk outlive the demo that produced them and usually cannot be
    rebuilt, so a field rename has to carry them forward rather than drop them.
    """
    data = dict(data)
    # Before there was a choice of engine, every session was a claude one.
    legacy = data.pop("session_id", None)
    if legacy and not data.get("sessions"):
        data["sessions"] = {"claude": legacy}
    # Builds before bilingual reports always produced Traditional Chinese.
    if data.get("language") not in LANGUAGES:
        data["language"] = "zh-TW"
    return data


class JobStore:
    """Jobs on disk under ``root/<job-id>/``, with an in-memory index.

    Persisted so a server restart does not throw away finished reports — the
    demo that produced them is usually deleted by then and cannot be re-parsed.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        for meta in sorted(self.root.glob("*/job.json")):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                job = Job(**_migrated(data))
            except (OSError, ValueError, TypeError) as exc:
                log.warning("skipping unreadable job %s: %s", meta.parent.name, exc)
                continue
            # A job that was mid-flight when the server died will never resume.
            if job.status in ("queued", "parsing", "coaching"):
                job.status = "failed"
                job.error = "server restarted while this job was running"
            self._jobs[job.id] = job

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create(
        self,
        *,
        original_name: str,
        player: str | None,
        coach: bool,
        keep_demo: bool,
        backend: str = DEFAULT_BACKEND,
        language: str = DEFAULT_LANGUAGE,
        analysis_request: str = "",
    ) -> Job:
        stem = safe_stem(original_name)
        job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        job = Job(
            id=job_id,
            stem=stem,
            original_name=original_name,
            created=_now(),
            player=player or None,
            backend=backend,
            language=language,
            analysis_request=analysis_request,
            coach=coach,
            keep_demo=keep_demo,
        )
        self.dir(job_id).mkdir(parents=True, exist_ok=True)
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        with self._lock:
            path = self.dir(job.id) / "job.json"
            # A user may delete a job while its subprocess is winding down.
            # Do not recreate that deleted job or crash the sole worker thread.
            if job.id not in self._jobs and not path.parent.exists():
                return
            self._jobs[job.id] = job
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if self._jobs.pop(job_id, None) is None:
                return False
        shutil.rmtree(self.dir(job_id), ignore_errors=True)
        return True


@dataclass
class Call:
    """One prepared model invocation: what to run, and where the answer lands."""

    argv: list[str]
    #: Known before the call only for engines that let the caller name the
    #: session. Otherwise it is read back out of the output afterwards.
    session_id: str | None = None
    #: Scratch file an engine writes its final message to, when its stdout also
    #: carries progress events and cannot be used as the answer.
    reply_file: Path | None = None


class Backend:
    """A CLI that can hold a conversation about one report.

    Everything model-shaped in this server goes through this interface: open a
    session with a prompt (the report arrives on stdin), resume it by id, and
    pull the reply back out. Two implementations exist so a quota that runs out
    on one side does not end the evening.
    """

    name = ""
    label = ""
    binary = ""

    @property
    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def open(self, prompt: str, *, work_dir: Path) -> Call:
        raise NotImplementedError

    def resume(self, session_id: str, message: str, *, work_dir: Path) -> Call:
        raise NotImplementedError

    def finish(self, call: Call, proc: subprocess.CompletedProcess[str]) -> tuple[str, str | None]:
        """``(reply, session id)``. An empty reply means the call failed."""
        raise NotImplementedError

    def failure(self, proc: subprocess.CompletedProcess[str]) -> str:
        return _tail(proc.stderr)


class ClaudeBackend(Backend):
    name, label, binary = "claude", "Claude", "claude"

    def open(self, prompt: str, *, work_dir: Path) -> Call:
        # Naming the session up front is what makes the review the *first turn*
        # of a conversation the user can carry on later, instead of a dead end.
        session_id = str(uuid.uuid4())
        return Call(["claude", "-p", "--session-id", session_id, prompt], session_id=session_id)

    def resume(self, session_id: str, message: str, *, work_dir: Path) -> Call:
        return Call(["claude", "-p", "--resume", session_id, message], session_id=session_id)

    def finish(self, call: Call, proc: subprocess.CompletedProcess[str]) -> tuple[str, str | None]:
        return (proc.stdout or "").strip(), call.session_id


class CodexBackend(Backend):
    """The same conversation, driven through ``codex exec``.

    Two differences from claude shape the code: the session id only exists once
    the run has started, so it is read off the JSONL event stream instead of
    chosen; and stdout carries those events, so the answer itself is taken from
    the file ``-o`` writes.
    """

    name, label, binary = "codex", "Codex", "codex"

    def open(self, prompt: str, *, work_dir: Path) -> Call:
        reply = self._reply_path(work_dir)
        argv = [
            "codex",
            "exec",
            "--json",
            # It only ever needs to *read* the match.json sitting next to it.
            "--sandbox",
            "read-only",
            # A job directory is not a repository, and codex declines to run in
            # one unless told that is expected.
            "--skip-git-repo-check",
            "-o",
            str(reply),
            prompt,
        ]
        return Call(argv, reply_file=reply)

    def resume(self, session_id: str, message: str, *, work_dir: Path) -> Call:
        reply = self._reply_path(work_dir)
        # `resume` accepts neither --sandbox nor --cd: it reuses the sandbox the
        # session was opened with, and the working directory it inherits from us.
        # Options also have to precede the session id, which is positional.
        argv = [
            "codex",
            "exec",
            "resume",
            "--json",
            "--skip-git-repo-check",
            "-o",
            str(reply),
            session_id,
            message,
        ]
        return Call(argv, session_id=session_id, reply_file=reply)

    def finish(self, call: Call, proc: subprocess.CompletedProcess[str]) -> tuple[str, str | None]:
        reply = ""
        if call.reply_file is not None and call.reply_file.exists():
            reply = call.reply_file.read_text(encoding="utf-8").strip()
            call.reply_file.unlink(missing_ok=True)
        return reply, call.session_id or codex_session(proc.stdout)

    def failure(self, proc: subprocess.CompletedProcess[str]) -> str:
        # codex reports most failures as events on stdout, so the stderr tail
        # alone would leave the page showing an empty error box.
        return _tail(proc.stderr) or _tail("\n".join(codex_errors(proc.stdout)))

    def _reply_path(self, work_dir: Path) -> Path:
        # Unique per call: a review and a question can be in flight at once.
        return work_dir / f".codex-reply-{uuid.uuid4().hex[:8]}.txt"


#: Every engine the page can offer, in the order it shows them.
BACKENDS: dict[str, Backend] = {b.name: b for b in (ClaudeBackend(), CodexBackend())}


def backend_for(name: str | None) -> Backend:
    """Resolve a stored or requested engine name, never raising on junk."""
    return BACKENDS.get(name or "", BACKENDS[DEFAULT_BACKEND])


def _codex_events(stdout: str | None):
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            yield json.loads(line)
        except ValueError:
            continue


def codex_session(stdout: str | None) -> str | None:
    """The thread id codex announces on its first event, or ``None``."""
    for event in _codex_events(stdout):
        if event.get("type") == "thread.started" and event.get("thread_id"):
            return str(event["thread_id"])
    return None


def codex_errors(stdout: str | None) -> list[str]:
    messages = []
    for event in _codex_events(stdout):
        item = event.get("item") or {}
        for candidate in (item, event):
            if candidate.get("type") == "error" and candidate.get("message"):
                messages.append(str(candidate["message"]))
    return messages


class Runner:
    """Serial worker: parse the demo, then coach the report."""

    def __init__(self, store: JobStore, *, coach_enabled: bool = True) -> None:
        self.store = store
        self.coach_enabled = coach_enabled
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="deadlock-coach-runner", daemon=True)
        self._thread.start()

    def submit(self, job_id: str, *, stage: str = "parse") -> None:
        self._queue.put((stage, job_id))

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def _loop(self) -> None:
        while True:
            stage, job_id = self._queue.get()
            job = self.store.get(job_id)
            if job is None:
                continue
            try:
                if stage == "parse":
                    self._parse_stage(job)
                else:
                    self._coach_stage(job)
            except Exception as exc:  # a worker thread that dies stops every later job
                log.exception("job %s crashed", job_id)
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished = _now()
                self.store.save(job)

    def _parse_stage(self, job: Job) -> None:
        """Decode the demo. Stops at the roster instead of guessing a perspective.

        The pipeline is run *without* ``--player`` on purpose: which hero the
        user played is not knowable until they say so, and the full report is
        what lets the review compare them against their own team afterwards.
        """
        job_dir = self.store.dir(job.id)
        demo_path = job_dir / f"{job.stem}.dem"
        job.started = _now()
        job.status = "parsing"
        job.note(f"parsing {job.original_name} ({job.size_bytes / 1e6:.0f} MB)")
        self.store.save(job)

        started = time.monotonic()
        argv = [sys.executable, "-m", "deadlock_coach.cli", str(demo_path), "-o", str(job_dir)]
        proc = self._exec(argv, timeout=PARSE_TIMEOUT_SECONDS, job=job)
        job.parse_seconds = round(time.monotonic() - started, 1)

        if proc.returncode != 0:
            job.status = "failed"
            job.error = _tail(proc.stderr) or f"deadlock-coach exited {proc.returncode}"
            job.finished = _now()
            self.store.save(job)
            return

        report = job_dir / f"{job.stem}.report.md"
        match_json = job_dir / f"{job.stem}.match.json"
        viewer_json = job_dir / f"{job.stem}.viewer.json"
        if not report.exists():
            job.status = "failed"
            job.error = "the pipeline reported success but wrote no report"
            job.finished = _now()
            self.store.save(job)
            return

        job.files["report"] = report.name
        if match_json.exists():
            job.files["json"] = match_json.name
            job.roster = read_roster(match_json)
        if viewer_json.exists():
            job.files["viewer"] = viewer_json.name
        summary_json = job_dir / f"{job.stem}.summary.json"
        if summary_json.exists():
            job.files["summary"] = summary_json.name
        job.note(f"parsed in {job.parse_seconds:.0f}s")
        timing = _pipeline_timing(proc.stderr)
        if timing:
            job.note(timing)

        # The demo is by far the biggest thing here and is never needed again
        # once the report exists, so drop it unless asked to keep it.
        if not job.keep_demo:
            demo_path.unlink(missing_ok=True)
            job.note("deleted the uploaded .dem")

        if not (job.coach and self.coach_enabled):
            job.status = "done"
            job.finished = _now()
            self.store.save(job)
            return

        # Hand back to the user: a review written for the wrong hero is worse
        # than no review, and the roster is only knowable once decoding is done.
        job.status = "awaiting_player"
        job.note("waiting for the user to pick a hero")
        self.store.save(job)

    def _coach_stage(self, job: Job) -> None:
        report = self.store.dir(job.id) / job.files.get("report", "")
        if not report.exists():
            job.status = "failed"
            job.error = "the report went missing before the review could run"
            self.store.save(job)
            return
        job.status = "coaching"
        self.store.save(job)
        self._coach(job, report)
        job.status = "done"
        job.finished = _now()
        self.store.save(job)

    def _coach(self, job: Job, report: Path) -> None:
        backend = backend_for(job.backend)
        prompt = PROMPT_PATHS[job.language].read_text(encoding="utf-8")
        if job.player:
            prompt += "\n\n" + perspective_line(job)
        if job.analysis_request:
            if job.language == "zh-TW":
                prompt += (
                    "\n\n## 使用者這次最想分析的問題\n"
                    + job.analysis_request
                    + "\n\n請優先、明確回答這個問題；仍須以報告中的實際證據為準。"
                )
            else:
                prompt += (
                    "\n\n## The user's requested focus\n"
                    + job.analysis_request
                    + "\n\nAnswer this explicitly and early, while staying within the report evidence."
                )

        work_dir = self.store.dir(job.id)
        call = backend.open(prompt, work_dir=work_dir)
        started = time.monotonic()
        model_input = report_for_model(report, job, work_dir) + visual_context_for_model(
            work_dir, job
        )
        proc = self._exec(
            call.argv,
            timeout=COACH_TIMEOUT_SECONDS,
            job=job,
            stdin_text=model_input,
            cwd=work_dir,
        )
        job.coach_seconds = round(time.monotonic() - started, 1)
        reply, session_id = backend.finish(call, proc)

        if proc.returncode != 0 or not reply:
            # The report is the valuable artifact and it already exists, so a
            # failed coaching pass is a warning on a finished job, not a failure.
            job.error = backend.failure(proc) or f"{backend.binary} produced no output"
            job.note(f"{backend.label} coaching failed; the report and JSON are still available")
            return

        path = work_dir / "coaching.md"
        path.write_text(reply, encoding="utf-8")
        job.files["coaching"] = path.name
        if session_id:
            job.sessions[backend.name] = session_id
        job.note(f"coached by {backend.label} in {job.coach_seconds:.0f}s")

    def _exec(
        self,
        argv: list[str],
        *,
        timeout: int,
        job: Job,
        stdin_text: str | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        log.info("job %s: %s", job.id, " ".join(argv[:3]))
        return run_process(argv, timeout=timeout, stdin_text=stdin_text, cwd=cwd)


class Chatter:
    """Follow-up questions, answered inside the session that wrote the review.

    Resuming replays the whole conversation, so the model still has the report
    in front of it and the user can just ask "那 15 分那波呢?" without
    re-uploading or re-explaining anything.

    Sessions belong to one engine and one working directory, so every call for a
    job runs from that job's directory — the same one ``Runner._coach`` used —
    and a question sent to the *other* engine opens a session of its own,
    carrying the report and the tail of the conversation in with it.
    """

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, job_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(job_id, threading.Lock())

    def _path(self, job_id: str) -> Path:
        return self.store.dir(job_id) / "chat.json"

    def history(self, job: Job) -> list[dict]:
        path = self._path(job.id)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return []

    def ask(self, job: Job, message: str, *, backend: str | None = None) -> dict:
        """Run one turn. Raises ``ChatError`` with something worth showing."""
        report = self.store.dir(job.id) / job.files.get("report", "")
        if not job.files.get("report") or not report.exists():
            raise ChatError(
                "這場還沒有報告，沒東西可以聊"
                if job.language == "zh-TW"
                else "This replay does not have a report to discuss yet."
            )

        engine = backend_for(backend or job.backend)
        with self._lock_for(job.id):
            work_dir = self.store.dir(job.id)
            known = job.sessions.get(engine.name)
            visual = visual_context_for_model(work_dir, job)
            if known:
                synced_message = message + visual
                call = engine.resume(known, synced_message, work_dir=work_dir)
                stdin_text = None
            else:
                # This engine has never seen the match: either nothing wrote a
                # review (--no-coach, or a failed pass) or the user just switched
                # sides. Both cases want the report in with the question, plus
                # whatever was already asked, so a switch continues the
                # conversation instead of restarting it.
                bootstrap = CHAT_PROMPT_PATHS[job.language].read_text(encoding="utf-8")
                if job.player:
                    bootstrap += "\n\n" + perspective_line(job)
                recap = handover(self.history(job), language=job.language)
                if recap:
                    bootstrap += "\n\n" + recap
                question_lead = (
                    "現在回答使用者的問題："
                    if job.language == "zh-TW"
                    else "Now answer the user's question:"
                )
                call = engine.open(f"{bootstrap}\n\n{question_lead}\n\n{message}", work_dir=work_dir)
                stdin_text = report_for_model(report, job, work_dir) + visual

            started = time.monotonic()
            proc = self._exec(
                call.argv,
                timeout=CHAT_TIMEOUT_SECONDS,
                stdin_text=stdin_text,
                cwd=work_dir,
            )
            elapsed = round(time.monotonic() - started, 1)

            reply, session_id = engine.finish(call, proc)
            if proc.returncode != 0 or not reply:
                fallback = (
                    f"{engine.label} 沒有回應"
                    if job.language == "zh-TW"
                    else f"{engine.label} did not return a response"
                )
                raise ChatError(engine.failure(proc) or fallback)

            if session_id:
                job.sessions[engine.name] = session_id
            job.backend = engine.name
            turns = self.history(job)
            turns.append({"role": "user", "text": message, "at": _now()})
            turns.append(
                {
                    "role": "assistant",
                    "text": reply,
                    "at": _now(),
                    "seconds": elapsed,
                    "backend": engine.name,
                }
            )
            del turns[: max(0, len(turns) - MAX_CHAT_TURNS * 2)]
            self._path(job.id).write_text(
                json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            job.chat_turns = len(turns)
            self.store.save(job)
            return {"reply": reply, "seconds": elapsed, "messages": turns}

    def _exec(
        self,
        argv: list[str],
        *,
        timeout: int,
        stdin_text: str | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return run_process(argv, timeout=timeout, stdin_text=stdin_text, cwd=cwd)


class ChatError(Exception):
    """Something the user should see in the chat box, not a stack trace."""


def run_process(
    argv: list[str],
    *,
    timeout: int,
    stdin_text: str | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, turning "missing" and "hung" into ordinary results."""
    log.info("exec %s", " ".join(argv[:4]))
    try:
        return subprocess.run(
            argv,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(argv, 127, "", f"{argv[0]} not found: {exc}")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 124, "", f"timed out after {timeout}s")


def _hero_names():
    """Cached ``hero_id -> name`` table. Degrades to bare ids without boon."""
    global _HERO_NAMES
    if _HERO_NAMES is None:
        from .names import Names

        _HERO_NAMES = Names.from_boon()
    return _HERO_NAMES


def read_roster(match_json: Path) -> list[dict]:
    """The twelve heroes of the match, for the user to pick themselves out of.

    Everything here comes from the sidecar the pipeline just wrote, so the
    picker cannot disagree with the report it is attached to.
    """
    try:
        data = json.loads(match_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("could not read the roster from %s: %s", match_json, exc)
        return []

    names = _hero_names()
    teams = {str(k): v for k, v in (data.get("match", {}).get("teams") or {}).items()}
    roster = []
    for player in data.get("players", []):
        hero_id = player.get("hero_id")
        if not hero_id:  # hero_id 0 is the environment, never a person
            continue
        roster.append(
            {
                "hero_id": hero_id,
                "hero": names.hero(hero_id),
                "player_name": player.get("player_name") or "",
                "team_num": player.get("team_num"),
                "team": teams.get(str(player.get("team_num")), "—"),
                "kda": f"{player.get('kills', 0)}/{player.get('deaths', 0)}"
                f"/{player.get('assists', 0)}",
                "net_worth": player.get("final_net_worth"),
            }
        )
    roster.sort(key=lambda r: (str(r["team_num"]), -(r["net_worth"] or 0)))
    return roster


def perspective_line(job: Job) -> str:
    """Tell the model exactly who the user is — no inference from the report."""
    entry = next((r for r in job.roster if r["hero"] == job.player), None)
    if job.language == "en":
        if entry is None:
            return f"The user wants the perspective of **{job.player}**. Center the review on that player."
        who = entry["hero"]
        if entry["player_name"]:
            who += f" (player name: {entry['player_name']})"
        return (
            f"The user played **{who}** on {entry['team']}. Center the review entirely on this "
            "player; mention others only when comparison or context requires it. Do not infer a "
            "different identity."
        )
    if entry is None:
        return f"這位使用者要看的是 **{job.player}** 的視角，複盤請以這位玩家為主。"
    who = entry["hero"]
    if entry["player_name"]:
        who += f"（玩家名 {entry['player_name']}）"
    return (
        f"使用者這場玩的是 **{who}**，{entry['team']} 隊。複盤請完全以這位玩家為主，"
        "其他人只在需要對照時提到。不要猜他是別人。"
    )


def handover(turns: list[dict], *, language: str = DEFAULT_LANGUAGE) -> str:
    """Replay the tail of the conversation for an engine that was not there.

    Switching engines mid-match is the whole point of having two, and a coach
    that has forgotten what you just told it is barely worth switching to.
    """
    recent = [t for t in turns if (t.get("text") or "").strip()][-HANDOVER_TURNS:]
    if not recent:
        return ""
    lines = [
        "這場之前已經聊過幾句（由另一個引擎回答），接著講就好，不用重新自我介紹："
        if language == "zh-TW"
        else "Another engine already handled these recent turns. Continue naturally without reintroducing yourself:"
    ]
    for turn in recent:
        if language == "zh-TW":
            who = "使用者" if turn.get("role") == "user" else "教練"
        else:
            who = "User" if turn.get("role") == "user" else "Coach"
        text = str(turn["text"]).strip()
        if len(text) > HANDOVER_CHARS:
            text = text[:HANDOVER_CHARS] + ("…（略）" if language == "zh-TW" else "… [trimmed]")
        lines.append(f"\n{who}{'：' if language == 'zh-TW' else ':'} {text}")
    return "\n".join(lines)


def report_for_model(report: Path, job: Job, work_dir: Path | None = None) -> str:
    text = report.read_text(encoding="utf-8")
    original_chars = len(text)
    if job.player:
        text = _focused_report(text, job)
        if len(text) < original_chars and not any(
            line.startswith("model input focused") for line in job.log
        ):
            job.note(
                f"model input focused from {original_chars / 1000:.0f} KB "
                f"to {len(text) / 1000:.0f} KB"
            )
    if len(text) > MAX_REPORT_CHARS:
        text = text[:MAX_REPORT_CHARS] + "\n\n[report truncated by the server]\n"
        job.note("report was unusually large and got truncated before the model saw it")
    # The saved report was rendered before anyone had said which hero they
    # played, so it carries no ranked summary. The one for the hero they did
    # pick is prepended here, ahead of the truncation cut, because the summary
    # is the part of the input the review is written from.
    lead = summary_for_model(work_dir, job) if work_dir is not None else ""
    return lead + text


def summary_for_model(work_dir: Path | None, job: Job) -> str:
    """The picked hero's ranked summary, as the report's opening section."""
    name = job.files.get("summary")
    if not name or not job.player or work_dir is None:
        return ""
    try:
        payload = json.loads((work_dir / name).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ""
    entry = next((row for row in job.roster if row.get("hero") == job.player), None)
    hero_id = entry.get("hero_id") if entry else None
    rows = payload.get("heroes") or []
    row = next(
        (r for r in rows if r.get("hero_id") == hero_id or r.get("hero") == job.player),
        None,
    )
    markdown = (row or {}).get("markdown")
    return f"{markdown}\n\n" if markdown else ""


def visual_context_for_model(work_dir: Path, job: Job) -> str:
    """Compact the visual sidecar into evidence a text-only model can use.

    The browser receives the full per-second payload.  Sending that multi-MB
    JSON to every model turn would be wasteful, so the model gets ten-second
    position samples plus every inventory and skill event for the picked hero.
    """
    name = job.files.get("viewer")
    if not name or not job.player:
        return ""
    try:
        data = json.loads((work_dir / name).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ""
    if not any(data.get(key) for key in ("positions", "inventory_events", "ability_upgrades", "ability_uses")):
        return ""

    roster_entry = next((row for row in job.roster if row.get("hero") == job.player), None)
    hero_id = roster_entry.get("hero_id") if roster_entry else None
    if hero_id is None:
        return ""

    clocks = data.get("clock") or []
    clock_ticks = [int(row[0]) for row in clocks]

    def clock_at(tick: object) -> str:
        value = int(tick or 0)
        index = bisect_right(clock_ticks, value) - 1
        seconds = float(clocks[index][1]) if index >= 0 else 0.0
        seconds = max(0, round(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    assets = data.get("assets") or {}
    zh = job.language == "zh-TW"

    def asset_name(asset_id: object) -> str:
        asset = assets.get(str(asset_id)) or {}
        if zh:
            localized = (asset.get("translations") or {}).get("zh-TW") or {}
            if localized.get("name"):
                return str(localized["name"])
        return str(asset.get("name") or asset.get("class_name") or f"#{asset_id}")

    if zh:
        lines = [
            "\n\n## 同步的視覺 replay 證據（viewer.json）",
            "以下資料和網頁戰術地圖來自同一份 sidecar。它是權威的裝備、技能與逐秒位置來源；",
            "不要聲稱報告沒有這些資料。文字複盤中的 MM:SS 會由網頁連回同一時間點。",
            f"檢視角色：{job.player}（hero_id {hero_id}）。",
            f"地圖圖片：{(data.get('map') or {}).get('image') or '離線'}",
            "位置以地圖左上為 (0%, 0%)；名稱使用 viewer 的繁中素材快照。",
            "",
            "### 裝備變更",
        ]
    else:
        lines = [
            "\n\n## Synchronized visual replay evidence (viewer.json)",
            "This data and the browser's tactical map come from the same sidecar. Treat it as the ",
            "authoritative source for inventory, abilities, and sampled positions; do not claim the ",
            "report lacks them. MM:SS timestamps in the review link back to the same moment.",
            f"Selected hero: {job.player} (hero_id {hero_id}).",
            f"Map image: {(data.get('map') or {}).get('image') or 'offline'}",
            "Map origin is (0%, 0%) at the upper left. Asset names use the English snapshot.",
            "",
            "### Inventory changes",
        ]
    inventory = [
        row for row in (data.get("inventory_events") or [])
        if int(row.get("hero_id") or -1) == int(hero_id)
    ]
    if inventory:
        for row in inventory:
            lines.append(
                f"- {clock_at(row.get('tick'))}{'：' if zh else ': '}{asset_name(row.get('ability_id'))} "
                f"({row.get('change') or 'changed'})"
            )
    else:
        lines.append("- 沒有記錄到裝備變更。" if zh else "- No inventory changes were recorded.")

    lines.extend(["", "### 技能升級" if zh else "### Ability upgrades"])
    upgrades = [
        row for row in (data.get("ability_upgrades") or [])
        if int(row.get("hero_id") or -1) == int(hero_id)
    ]
    if upgrades:
        for row in upgrades:
            lines.append(
                f"- {clock_at(row.get('tick'))}{'：' if zh else ': '}{asset_name(row.get('ability_id'))} "
                f"→ T{row.get('tier')}"
            )
    else:
        lines.append("- 沒有記錄到技能升級。" if zh else "- No ability upgrades were recorded.")

    uses: dict[str, list[str]] = {}
    by_class = {}
    for asset in assets.values():
        if not asset.get("class_name"):
            continue
        localized = (asset.get("translations") or {}).get("zh-TW") or {}
        display_name = localized.get("name") if zh else asset.get("name")
        by_class[str(asset["class_name"])] = str(display_name or asset["class_name"])
    for row in data.get("ability_uses") or []:
        if int(row.get("hero_id") or -1) != int(hero_id):
            continue
        label = by_class.get(
            str(row.get("ability")),
            str(row.get("ability") or ("未知技能" if zh else "unknown ability")),
        )
        uses.setdefault(label, []).append(clock_at(row.get("tick")))
    lines.extend(["", "### 技能使用時間" if zh else "### Ability-use timestamps"])
    if uses:
        for label, times in uses.items():
            shown = times[:80]
            suffix = (
                (f"（另有 {len(times) - len(shown)} 次）" if zh else f" ({len(times) - len(shown)} more)")
                if len(times) > len(shown)
                else ""
            )
            lines.append(f"- {label}{'：' if zh else ': '}{', '.join(shown)}{suffix}")
    else:
        lines.append("- 沒有記錄到技能使用事件。" if zh else "- No ability-use events were recorded.")

    map_data = data.get("map") or {}
    radius = float(map_data.get("radius") or 10_752)
    objectives = []
    for objective, point in (map_data.get("objective_positions") or {}).items():
        try:
            x = float(point["left_relative"]) * 2 * radius - radius
            y = radius - float(point["top_relative"]) * 2 * radius
        except (KeyError, TypeError, ValueError):
            continue
        objectives.append((str(objective), x, y))

    lines.extend(["", "### 每 10 秒位置取樣" if zh else "### Position samples every 10 seconds"])
    last_bucket = -1
    for row in data.get("positions") or []:
        if len(row) < 6 or int(row[1]) != int(hero_id):
            continue
        index = bisect_right(clock_ticks, int(row[0])) - 1
        seconds = int(float(clocks[index][1])) if index >= 0 else 0
        bucket = seconds // 10
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        x, y = float(row[2]), float(row[3])
        left = max(0.0, min(100.0, (x + radius) / (2 * radius) * 100))
        top = max(0.0, min(100.0, (radius - y) / (2 * radius) * 100))
        nearest = ""
        if objectives:
            objective, ox, oy = min(objectives, key=lambda item: (x - item[1]) ** 2 + (y - item[2]) ** 2)
            distance = round(((x - ox) ** 2 + (y - oy) ** 2) ** 0.5)
            nearest = (
                f"，最近目標 {objective}（{distance} units）"
                if zh
                else f", nearest objective {objective} ({distance} units)"
            )
        hp = ("，" if zh else ", ") + f"HP {row[6]}/{row[7]}" if len(row) > 7 and row[7] else ""
        alive = ("存活" if bool(row[5]) else "死亡／最後位置") if zh else (
            "alive" if bool(row[5]) else "dead / last position"
        )
        if zh:
            lines.append(f"- {clock_at(row[0])}：地圖 left {left:.1f}%, top {top:.1f}%{nearest}{hp}，{alive}")
        else:
            lines.append(f"- {clock_at(row[0])}: map left {left:.1f}%, top {top:.1f}%{nearest}{hp}, {alive}")

    context = "\n".join(lines) + "\n"
    if len(context) > MAX_VISUAL_CONTEXT_CHARS:
        suffix = "[視覺上下文因長度上限截斷]" if zh else "[visual context truncated at the size limit]"
        context = context[:MAX_VISUAL_CONTEXT_CHARS] + f"\n{suffix}\n"
    return context


def _pipeline_timing(stderr: str | None) -> str:
    """Extract the CLI's stable timing summary from otherwise noisy logs."""
    marker = "pipeline timing:"
    for line in reversed((stderr or "").splitlines()):
        if marker in line:
            return line[line.index(marker) :].strip()
    return ""


def _focused_report(text: str, job: Job) -> str:
    """Remove other players' verbose detail before the first coaching call.

    The roster, phases, economy, advantage ledger, and teamfight table remain
    whole so the model can still explain the shape of the match.  Only sections
    explicitly repeated once per player and the raw event timeline are reduced.
    The report saved for download is never changed.
    """
    if not job.player or "\n## " not in text:
        return text

    entry = next((row for row in job.roster if row.get("hero") == job.player), None)
    labels = [job.player]
    if entry and entry.get("player_name"):
        labels.append(str(entry["player_name"]))

    blocks = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    focused: list[str] = []
    for block in blocks:
        heading = block.splitlines()[0].strip() if block.strip() else ""
        if heading in ("## Player-perspective opportunities", "## Per-player review"):
            block = _player_subsection(block, job.player)
        elif heading == "## Event timeline":
            block = _focused_timeline(block, labels)
        focused.append(block)

    result = "".join(focused)
    note = (
        f"\n\n[Server note: verbose per-player tables and the raw timeline were focused on "
        f"{job.player}; global match, economy, advantage, and teamfight sections are complete.]\n"
    )
    return result.rstrip() + note


def _player_subsection(block: str, hero: str) -> str:
    """Keep a section's shared explanation plus the selected hero's ``###`` block."""
    chunks = re.split(r"(?=^### )", block, flags=re.MULTILINE)
    if len(chunks) == 1:
        return block
    wanted = hero.casefold()
    matches = [
        chunk
        for chunk in chunks[1:]
        if chunk.splitlines()[0].removeprefix("### ").casefold().startswith(wanted + " ")
    ]
    # A report-format/name mismatch must cost speed, not evidence.
    if not matches:
        return block
    return chunks[0].rstrip() + "\n\n" + "\n\n".join(chunk.rstrip() for chunk in matches) + "\n"


def _focused_timeline(block: str, labels: list[str]) -> str:
    """Keep selected-player and public-objective events from the raw timeline."""
    needles = tuple(label.casefold() for label in labels if label)
    kept: list[str] = []
    event_count = 0
    omitted = 0
    for line in block.splitlines():
        if not line.startswith("["):
            kept.append(line)
            continue
        folded = line.casefold()
        relevant = any(needle in folded for needle in needles) or any(
            term in folded for term in _PUBLIC_EVENT_TERMS
        )
        if relevant and event_count < FOCUSED_MAX_TIMELINE_EVENTS:
            kept.append(line)
            event_count += 1
        else:
            omitted += 1

    if omitted:
        # Insert before the closing code fence so the note remains visibly part
        # of the event stream instead of looking like analysis evidence.
        insertion = f"[{omitted} unrelated timeline events omitted for coaching speed]"
        try:
            closing = len(kept) - 1 - kept[::-1].index("```")
        except ValueError:
            kept.append(insertion)
        else:
            kept.insert(closing, insertion)
    return "\n".join(kept).rstrip() + "\n"


def _tail(text: str | None, lines: int = 12) -> str:
    if not text:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])


def lan_address() -> str:
    """Best guess at the address another laptop on this WiFi should open."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packets are sent; this just asks the routing table which local
        # interface would be used to reach the internet.
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def create_app(*, data_dir: Path, coach_enabled: bool = True, default_backend: str | None = None):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    store = JobStore(data_dir / "jobs")
    runner = Runner(store, coach_enabled=coach_enabled)
    chatter = Chatter(store)
    fallback = default_backend if default_backend in BACKENDS else DEFAULT_BACKEND
    app = FastAPI(title="deadlock-replay-coach", docs_url=None, redoc_url=None)

    def _job_or_404(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job

    def _backend_or_400(value, current: str) -> str:
        """Validate against the *known* engines, not the installed ones.

        A missing binary is reported by the job that tried to use it, which says
        far more than a form rejecting a name the page just offered.
        """
        if value in (None, ""):
            return current
        name = str(value)
        if name not in BACKENDS:
            raise HTTPException(status_code=400, detail=f"unknown engine {name}")
        return name

    def _language_or_400(value, current: str = DEFAULT_LANGUAGE) -> str:
        if value in (None, ""):
            return current
        language = str(value)
        if language not in LANGUAGES:
            raise HTTPException(status_code=400, detail=f"unsupported language {language}")
        return language

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        # The whole frontend is this one file, JS inlined. With no validator a
        # browser caches it heuristically, so a phone that opened the page
        # yesterday keeps running yesterday's JS against today's API — which
        # looks like "the page is broken", not "the page is stale".
        return HTMLResponse(
            (WEB_DIR / "index.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/config")
    def config() -> dict:
        return {
            "coach_enabled": coach_enabled,
            "pending": runner.pending,
            # `available` is re-checked per request so installing the other CLI
            # only needs a page reload, not a server restart.
            "backends": [
                {"name": b.name, "label": b.label, "available": b.available}
                for b in BACKENDS.values()
            ],
            "default_backend": fallback,
        }

    @app.post("/api/upload")
    async def upload(
        demo: UploadFile = File(...),
        coach: str = Form("true"),
        keep_demo: str = Form("false"),
        backend: str = Form(""),
        language: str = Form(DEFAULT_LANGUAGE),
        analysis_request: str = Form(""),
    ) -> JSONResponse:
        # No perspective is accepted here on purpose — see `_parse_stage`. The
        # user picks a hero off the decoded roster, once there is one to show.
        if not (demo.filename or "").lower().endswith(".dem"):
            raise HTTPException(status_code=400, detail="expected a .dem replay")

        job = store.create(
            original_name=demo.filename or "replay.dem",
            player=None,
            coach=_flag(coach),
            keep_demo=_flag(keep_demo),
            backend=_backend_or_400(backend, fallback),
            language=_language_or_400(language),
            analysis_request=_analysis_request(analysis_request),
        )
        target = store.dir(job.id) / f"{job.stem}.dem"
        size = 0
        too_large = False
        try:
            with target.open("wb") as fh:
                while chunk := await demo.read(UPLOAD_CHUNK):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        too_large = True
                        break
                    fh.write(chunk)
        except OSError as exc:
            store.delete(job.id)
            raise HTTPException(status_code=500, detail=f"could not save upload: {exc}") from exc

        if too_large:
            store.delete(job.id)
            limit_gib = MAX_UPLOAD_BYTES / (1024**3)
            raise HTTPException(status_code=413, detail=f"replay exceeds the {limit_gib:g} GiB upload limit")

        if size == 0:
            store.delete(job.id)
            raise HTTPException(status_code=400, detail="the upload was empty")

        job.size_bytes = size
        job.note(f"received {size / 1e6:.0f} MB")
        store.save(job)
        runner.submit(job.id)
        return JSONResponse({"id": job.id})

    @app.get("/api/jobs")
    def jobs() -> dict:
        return {"jobs": [asdict(job) for job in store.all()], "pending": runner.pending}

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict:
        return asdict(_job_or_404(job_id))

    @app.get("/api/jobs/{job_id}/text/{kind}")
    def job_text(job_id: str, kind: str) -> dict:
        """Inline content for the browser to render, rather than download."""
        job = _job_or_404(job_id)
        name = job.files.get(kind)
        if not name:
            raise HTTPException(status_code=404, detail=f"job has no {kind}")
        return {"text": (store.dir(job_id) / name).read_text(encoding="utf-8")}

    @app.get("/api/jobs/{job_id}/viewer")
    def job_viewer(job_id: str):
        """Serve the visual replay, filling static map layers for old jobs."""
        job = _job_or_404(job_id)
        name = job.files.get("viewer")
        if not name:
            raise HTTPException(status_code=404, detail="job has no visual replay")
        path = store.dir(job_id) / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict) and int(payload.get("schema_version") or 0) < 2:
            from .viewer import upgrade_viewer_payload

            return JSONResponse(
                upgrade_viewer_payload(payload),
                headers={"Cache-Control": "private, max-age=31536000, immutable"},
            )
        return FileResponse(
            path,
            media_type="application/json",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    @app.get("/api/jobs/{job_id}/file/{kind}")
    def job_file(job_id: str, kind: str) -> FileResponse:
        job = _job_or_404(job_id)
        name = job.files.get(kind)
        if not name:
            raise HTTPException(status_code=404, detail=f"job has no {kind}")
        # The pipeline's own files already carry the stem; only coaching.md needs it
        # added, so that several downloads do not all land as "coaching.md".
        download = name if name.startswith(job.stem) else f"{job.stem}.{name}"
        return FileResponse(store.dir(job_id) / name, filename=download)

    @app.post("/api/jobs/{job_id}/player")
    def choose_player(job_id: str, body: dict) -> dict:
        """Pick the hero the user actually played, then start the review."""
        job = _job_or_404(job_id)
        if job.status != "awaiting_player":
            detail = (
                f"這場的狀態是 {job.status}，不能選角色"
                if job.language == "zh-TW"
                else f"cannot choose a hero while this replay is {job.status}"
            )
            raise HTTPException(status_code=409, detail=detail)

        # The pick is the last moment before the expensive call, so it is also
        # the right moment to change which engine pays for it.
        job.backend = _backend_or_400(body.get("backend"), job.backend)
        job.language = _language_or_400(body.get("language"), job.language)
        if "analysis_request" in body:
            job.analysis_request = _analysis_request(body.get("analysis_request"))

        hero_id = body.get("hero_id")
        if hero_id is None:
            job.player = None  # a whole-match review, with no one's perspective
            job.note("no hero picked; reviewing the match as a whole")
        else:
            entry = next((r for r in job.roster if r["hero_id"] == hero_id), None)
            if entry is None:
                detail = "這場沒有這個英雄" if job.language == "zh-TW" else "hero not found in this replay"
                raise HTTPException(status_code=400, detail=detail)
            job.player = entry["hero"]
            job.note(f"perspective: {entry['hero']} ({entry['player_name'] or '—'})")

        job.status = "queued"
        store.save(job)
        runner.submit(job.id, stage="coach")
        return {"player": job.player, "backend": job.backend}

    @app.get("/api/jobs/{job_id}/chat")
    def chat_history(job_id: str) -> dict:
        job = _job_or_404(job_id)
        return {"messages": chatter.history(job), "can_chat": bool(job.files.get("report"))}

    @app.post("/api/jobs/{job_id}/chat")
    def chat(job_id: str, body: dict) -> dict:
        """One follow-up turn. Runs in FastAPI's threadpool, not the parse queue,
        so asking about an old match never waits behind a demo that is decoding."""
        job = _job_or_404(job_id)
        if not coach_enabled:
            detail = (
                "這台 server 是用 --no-coach 開的"
                if job.language == "zh-TW"
                else "this server was started with --no-coach"
            )
            raise HTTPException(status_code=409, detail=detail)
        message = str(body.get("message", "")).strip()
        if not message:
            detail = "訊息是空的" if job.language == "zh-TW" else "message is empty"
            raise HTTPException(status_code=400, detail=detail)
        backend = _backend_or_400(body.get("backend"), job.backend)
        try:
            return chatter.ask(job, message, backend=backend)
        except ChatError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        if not store.delete(job_id):
            raise HTTPException(status_code=404, detail="no such job")
        return {"deleted": job_id}

    return app


def _flag(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "on", "yes")


def _analysis_request(value: object, *, limit: int = 2_000) -> str:
    """Normalize the optional focus without allowing unbounded persisted input."""
    return str(value or "").strip()[:limit]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="deadlock-coach-server",
        description="Upload a .dem from any device on the LAN and get a coached review back.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("server-data"),
        help="where uploads and reports live (default: ./server-data)",
    )
    parser.add_argument(
        "--no-coach",
        action="store_true",
        help="only decode; skip the model review pass entirely",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        help="which CLI writes the review by default (the page can override it per job)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    installed = [b for b in BACKENDS.values() if b.available]
    if not args.no_coach and not installed:
        log.warning("neither claude nor codex is on PATH; starting with coaching disabled")
        args.no_coach = True

    # Preferring an installed engine matters: the page offers both either way,
    # but the default should be one that can actually answer.
    default_backend = args.backend or (installed[0].name if installed else DEFAULT_BACKEND)

    app = create_app(
        data_dir=args.data_dir.resolve(),
        coach_enabled=not args.no_coach,
        default_backend=default_backend,
    )

    import uvicorn

    if args.host in ("0.0.0.0", "::"):
        # flush: this is the one thing the user needs off the screen, and it would
        # otherwise sit in a buffer until uvicorn logs something.
        print(f"\n  Upload page: http://{lan_address()}:{args.port}  (available on this LAN)", flush=True)
        print(f"  This machine: http://localhost:{args.port}", flush=True)
        if not args.no_coach:
            engines = ", ".join(
                f"{b.label}{' (default)' if b.name == default_backend else ''}" for b in installed
            )
            print(f"  Available engines: {engines} (switch anytime in the web UI)\n", flush=True)
        else:
            print("  Decode only; AI coaching is disabled\n", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
