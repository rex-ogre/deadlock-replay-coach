"""Server tests that never parse a demo or call a model.

Both heavy steps are subprocesses behind ``Runner._exec``, so stubbing that one
method exercises the whole job lifecycle — queueing, file discovery, demo
cleanup, failure handling — in milliseconds.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from deadlock_coach.server import (
    BACKENDS,
    ChatError,
    Chatter,
    Job,
    JobStore,
    Runner,
    _pipeline_timing,
    codex_session,
    handover,
    perspective_line,
    report_for_model,
    safe_stem,
    visual_context_for_model,
)

#: A stand-in sidecar. `hero_id` 0 is in there on purpose: it is what the demo
#: reports for trooper/objective kills, and it must never reach the picker.
MATCH_JSON = {
    "match": {"teams": {"2": "Hidden King", "3": "Archmother"}},
    "players": [
        {"hero_id": 12, "player_name": "Astraia", "team_num": 2,
         "kills": 2, "deaths": 2, "assists": 9, "final_net_worth": 32159},
        {"hero_id": 31, "player_name": "pioneer", "team_num": 2,
         "kills": 6, "deaths": 4, "assists": 5, "final_net_worth": 40147},
        {"hero_id": 4, "player_name": "PW", "team_num": 3,
         "kills": 10, "deaths": 0, "assists": 12, "final_net_worth": 56109},
        {"hero_id": 0, "player_name": "", "team_num": 2},
    ],
}

fastapi = pytest.importorskip("fastapi", reason="needs the `server` extra")


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("12345678.dem", "12345678"),
        ("../../etc/passwd.dem", "passwd"),  # only the basename survives
        ("a/b/../c.dem", "c"),
        ("比賽 99.dem", "比賽_99"),  # `\w` is unicode, so a Chinese name is kept
        ("$(rm -rf ~).dem", "rm_-rf"),
        ("", "replay"),
        ("...", "replay"),
    ],
)
def test_safe_stem_cannot_escape_the_job_directory(filename, expected):
    stem = safe_stem(filename)
    assert stem == expected
    assert "/" not in stem and ".." not in stem


def test_store_reloads_jobs_from_disk(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(original_name="12345678.dem", player="Yamato", coach=True, keep_demo=False)
    job.status = "done"
    job.files["report"] = "12345678.report.md"
    store.save(job)

    reloaded = JobStore(tmp_path).get(job.id)
    assert reloaded is not None
    assert reloaded.player == "Yamato"
    assert reloaded.language == "en"
    assert reloaded.files["report"] == "12345678.report.md"


def test_restart_fails_jobs_that_were_mid_flight(tmp_path):
    """Nothing resumes a job on restart, so leaving one "parsing" would hang the UI."""
    store = JobStore(tmp_path)
    for status in ("queued", "parsing", "coaching"):
        job = store.create(original_name=f"{status}.dem", player=None, coach=True, keep_demo=False)
        job.status = status
        store.save(job)

    revived = JobStore(tmp_path)
    assert {job.status for job in revived.all()} == {"failed"}
    assert all("restarted" in (job.error or "") for job in revived.all())


def test_a_late_worker_save_does_not_recreate_a_deleted_job(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(original_name="x.dem", player=None, coach=True, keep_demo=False)

    assert store.delete(job.id)
    job.status = "failed"
    store.save(job)

    assert store.get(job.id) is None
    assert not store.dir(job.id).exists()


def test_a_restart_leaves_a_job_that_is_only_waiting_on_the_user(tmp_path):
    """`awaiting_player` is not in-flight — nothing is running, so nothing is lost."""
    store = JobStore(tmp_path)
    job = store.create(original_name="x.dem", player=None, coach=True, keep_demo=False)
    job.status = "awaiting_player"
    job.roster = [{"hero_id": 12, "hero": "Ivy", "player_name": "a", "team_num": 2,
                   "team": "Hidden King", "kda": "1/2/3", "net_worth": 1}]
    store.save(job)

    reloaded = JobStore(tmp_path).get(job.id)
    assert reloaded.status == "awaiting_player"
    assert reloaded.error is None
    assert reloaded.roster[0]["hero"] == "Ivy"


#: The id codex announces on its first event; claude's is chosen by the caller.
CODEX_THREAD = "01a0057d-d299-79d3-a9be-237233d38d1a"


def _fake_model(argv, out, rc):
    """Answer like the CLI in ``argv[0]`` would, including where it writes.

    codex prints JSONL progress on stdout and puts the reply in the file named
    by ``-o``, so a stub that only returns stdout would let a real regression in
    that plumbing pass the tests.
    """
    if argv[0] != "codex":
        return subprocess.CompletedProcess(argv, rc, out, "boom")
    if rc == 0 and out:
        Path(argv[argv.index("-o") + 1]).write_text(out, encoding="utf-8")
    events = [
        json.dumps({"type": "thread.started", "thread_id": CODEX_THREAD}),
        json.dumps({"type": "item.completed", "item": {"type": "error", "message": "boom"}}),
    ]
    return subprocess.CompletedProcess(argv, rc, "\n".join(events) + "\n", "")


class StubRunner(Runner):
    """A runner whose subprocesses are scripted instead of real."""

    def __init__(self, store, *, parse_rc=0, coach_rc=0, coach_out="# 複盤\n- 一件事\n"):
        self.calls: list[list[str]] = []
        self.parse_rc = parse_rc
        self.coach_rc = coach_rc
        self.coach_out = coach_out
        super().__init__(store, coach_enabled=True)

    def _exec(self, argv, *, timeout, job, stdin_text=None, cwd=None):
        self.calls.append(argv)
        if argv[0] in BACKENDS:
            self.coach_stdin = stdin_text
            return _fake_model(argv, self.coach_out, self.coach_rc)
        if self.parse_rc == 0:
            out = self.store.dir(job.id)
            (out / f"{job.stem}.report.md").write_text("# report", encoding="utf-8")
            (out / f"{job.stem}.match.json").write_text(
                json.dumps(MATCH_JSON), encoding="utf-8"
            )
            (out / f"{job.stem}.viewer.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
        return subprocess.CompletedProcess(argv, self.parse_rc, "", "parse failed: bad magic")


def _job_with_demo(store, **kwargs):
    kwargs.setdefault("original_name", "12345678.dem")
    kwargs.setdefault("player", None)
    kwargs.setdefault("coach", True)
    kwargs.setdefault("keep_demo", False)
    job = store.create(**kwargs)
    (store.dir(job.id) / f"{job.stem}.dem").write_bytes(b"PBDEMS2")
    return job


def _decode(store, runner, **kwargs):
    """Upload and decode, stopping where the user has to point at themselves."""
    job = _job_with_demo(store, **kwargs)
    runner._parse_stage(job)
    return job


def _decode_and_pick(store, runner, hero_id=12, **kwargs):
    job = _decode(store, runner, **kwargs)
    if job.status == "awaiting_player":
        job.player = next(r["hero"] for r in job.roster if r["hero_id"] == hero_id)
        runner._coach_stage(job)
    return job


def test_decoding_stops_at_the_roster_instead_of_guessing(tmp_path):
    """The bug this exists to prevent: a review written for the wrong hero."""
    store = JobStore(tmp_path)
    runner = StubRunner(store)

    job = _decode(store, runner)

    assert job.status == "awaiting_player"
    assert not any(argv[0] in BACKENDS for argv in runner.calls), "no review before the pick"
    assert "--player" not in runner.calls[0], "the pipeline cannot know who the user is"
    assert [r["hero_id"] for r in job.roster] == [31, 12, 4]  # by team, richest first
    assert [r["team"] for r in job.roster] == ["Hidden King", "Hidden King", "Archmother"]
    assert job.roster[1]["kda"] == "2/2/9"
    assert 0 not in [r["hero_id"] for r in job.roster], "hero_id 0 is not a person"


def test_picking_a_hero_produces_the_review(tmp_path):
    store = JobStore(tmp_path)
    runner = StubRunner(store)

    job = _decode_and_pick(store, runner)

    assert job.status == "done"
    assert job.error is None
    assert job.files == {
        "report": "12345678.report.md",
        "json": "12345678.match.json",
        "viewer": "12345678.viewer.json",
        "coaching": "coaching.md",
    }
    assert (store.dir(job.id) / "coaching.md").read_text(encoding="utf-8").startswith("# 複盤")
    assert runner.coach_stdin == "# report"  # the full report, not a trimmed one


def test_the_review_prompt_names_the_player_outright(tmp_path):
    """Perspective is stated, never inferred from the report."""
    store = JobStore(tmp_path)
    runner = StubRunner(store)
    job = _decode(store, runner)
    astraia = next(r for r in job.roster if r["hero_id"] == 12)
    job.player = astraia["hero"]

    line = perspective_line(job)
    assert astraia["hero"] in line
    assert "Astraia" in line and "Hidden King" in line
    assert "Do not infer a different identity" in line


def test_chinese_review_prompt_keeps_the_previous_perspective_wording(tmp_path):
    store = JobStore(tmp_path)
    job = _decode(store, StubRunner(store), language="zh-TW")
    job.player = next(r["hero"] for r in job.roster if r["hero_id"] == 12)

    assert "不要猜" in perspective_line(job)


def test_the_review_prioritizes_the_users_analysis_request(tmp_path):
    store = JobStore(tmp_path)
    runner = StubRunner(store)
    focus = "請分析 12:44 的站位、當時裝備與技能。"
    job = _decode(store, runner, analysis_request=focus)
    job.player = next(r["hero"] for r in job.roster if r["hero_id"] == 12)

    runner._coach_stage(job)

    assert focus in runner.calls[-1][-1]
    assert "Answer this explicitly and early" in runner.calls[-1][-1]


def test_chinese_review_uses_the_traditional_chinese_prompt(tmp_path):
    store = JobStore(tmp_path)
    runner = StubRunner(store)
    job = _decode(store, runner, language="zh-TW", analysis_request="請看站位")
    job.player = next(r["hero"] for r in job.roster if r["hero_id"] == 12)

    runner._coach_stage(job)

    assert "繁體中文" in runner.calls[-1][-1]
    assert "優先、明確回答" in runner.calls[-1][-1]


def test_visual_context_syncs_position_items_and_skills_for_the_model(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(
        original_name="map.dem", player="Ivy", coach=True, keep_demo=False
    )
    job.roster = [{"hero_id": 12, "hero": "Ivy"}]
    job.files["viewer"] = "map.viewer.json"
    payload = {
        "clock": [[0, 0.0], [640, 10.0]],
        "map": {
            "radius": 100,
            "image": "https://assets.example/map.png",
            "objective_positions": {
                "walker": {"left_relative": 0.5, "top_relative": 0.5}
            },
        },
        "positions": [[640, 12, 10, 20, 0, True, 500, 700]],
        "assets": {
            "101": {"name": "疾速彈", "class_name": "item_fast"},
            "201": {"name": "藤蔓", "class_name": "ability_vine"},
        },
        "inventory_events": [
            {"tick": 640, "hero_id": 12, "ability_id": 101, "change": "purchased"}
        ],
        "ability_upgrades": [
            {"tick": 640, "hero_id": 12, "ability_id": 201, "tier": 2}
        ],
        "ability_uses": [{"tick": 640, "hero_id": 12, "ability": "ability_vine"}],
    }
    (store.dir(job.id) / "map.viewer.json").write_text(json.dumps(payload), encoding="utf-8")

    context = visual_context_for_model(store.dir(job.id), job)

    assert "do not claim" in context
    assert "10:00" not in context
    assert "00:10: 疾速彈 (purchased)" in context
    assert "00:10: 藤蔓 → T2" in context
    assert "left 55.0%, top 40.0%" in context

    job.language = "zh-TW"
    chinese = visual_context_for_model(store.dir(job.id), job)
    assert "不要聲稱報告沒有這些資料" in chinese
    assert "00:10：疾速彈 (purchased)" in chinese


def test_player_review_sends_global_context_but_only_that_players_detail(tmp_path):
    report = tmp_path / "match.report.md"
    report.write_text(
        """# Replay briefing

## Roster and final line

| Yamato | SamplePlayer |
| Seven | Astraia |

## Teamfights

Seven won the decisive fight against Yamato's team.

## Player-perspective opportunities

Shared detector explanation.

| Player | Signals |
| --- | --- |
| Yamato (SamplePlayer) | 3 |
| Seven (Astraia) | 9 |

### Seven — every window, ranked by importance (1)

Seven-only opportunity detail.

### Yamato — every window, ranked by importance (1)

Yamato-only opportunity detail.

## Per-player review

### Seven — Astraia

Seven-only player review.

### Yamato — SamplePlayer

Yamato-only player review.

## Event timeline

```
[01:00] Seven killed somebody unrelated
[02:00] Yamato killed Seven
[03:00] Mid Boss spawned
```
""",
        encoding="utf-8",
    )
    job = Job(
        id="x",
        stem="match",
        original_name="match.dem",
        created="now",
        player="Yamato",
        roster=[{"hero": "Yamato", "player_name": "SamplePlayer"}],
    )

    focused = report_for_model(report, job)

    assert "Seven won the decisive fight" in focused  # global evidence stays whole
    assert "Yamato-only opportunity detail" in focused
    assert "Yamato-only player review" in focused
    assert "Seven-only opportunity detail" not in focused
    assert "Seven-only player review" not in focused
    assert "[02:00] Yamato killed Seven" in focused
    assert "[03:00] Mid Boss spawned" in focused
    assert "[01:00] Seven killed somebody unrelated" not in focused
    assert "unrelated timeline events omitted" in focused


def test_the_picked_heros_summary_leads_the_model_input(tmp_path):
    """The saved report predates the hero pick, so the summary is prepended."""
    report = tmp_path / "match.report.md"
    report.write_text("## Match\n\nbody\n", encoding="utf-8")
    (tmp_path / "match.summary.json").write_text(
        json.dumps(
            {
                "heroes": [
                    {"hero_id": 1, "hero": "Seven", "markdown": "## Bottom line\n\nSeven."},
                    {"hero_id": 2, "hero": "Yamato", "markdown": "## Bottom line\n\nYamato."},
                ]
            }
        ),
        encoding="utf-8",
    )
    job = Job(
        id="x",
        stem="match",
        original_name="match.dem",
        created="now",
        player="Yamato",
        roster=[{"hero": "Yamato", "hero_id": 2, "player_name": "SamplePlayer"}],
        files={"report": "match.report.md", "summary": "match.summary.json"},
    )

    text = report_for_model(report, job, tmp_path)

    assert text.startswith("## Bottom line")
    assert "Yamato." in text and "Seven." not in text
    assert text.index("## Bottom line") < text.index("## Match")


def test_model_input_survives_a_missing_summary(tmp_path):
    report = tmp_path / "match.report.md"
    report.write_text("## Match\n\nbody\n", encoding="utf-8")
    job = Job(
        id="x",
        stem="match",
        original_name="match.dem",
        created="now",
        player="Yamato",
        roster=[{"hero": "Yamato", "hero_id": 2, "player_name": "SamplePlayer"}],
        files={"report": "match.report.md"},
    )
    assert report_for_model(report, job, tmp_path).startswith("## Match")


def test_pipeline_timing_is_extracted_from_cli_noise():
    stderr = (
        "INFO parsing match.dem\n"
        "INFO pipeline timing: decode 12.0s, APIs 0.2s, analysis 1.3s, "
        "render 0.4s, total 13.9s\n"
    )
    assert _pipeline_timing(stderr) == (
        "pipeline timing: decode 12.0s, APIs 0.2s, analysis 1.3s, "
        "render 0.4s, total 13.9s"
    )


def test_demo_is_deleted_unless_the_upload_asked_to_keep_it(tmp_path):
    store = JobStore(tmp_path)
    runner = StubRunner(store)

    dropped = _decode(store, runner)
    assert not (store.dir(dropped.id) / "12345678.dem").exists()

    kept = _decode(store, runner, keep_demo=True)
    assert (store.dir(kept.id) / "12345678.dem").exists()


def test_a_failed_parse_reports_the_stderr_tail(tmp_path):
    store = JobStore(tmp_path)
    runner = StubRunner(store, parse_rc=2)
    job = _decode(store, runner)

    assert job.status == "failed"
    assert "bad magic" in job.error
    assert job.files == {}
    assert len(runner.calls) == 1  # coaching never ran


def test_a_failed_coach_still_leaves_a_usable_job(tmp_path):
    """The report is the artifact worth keeping; a dead model call is a warning."""
    store = JobStore(tmp_path)
    runner = StubRunner(store, coach_rc=1, coach_out="")
    job = _decode_and_pick(store, runner)

    assert job.status == "done"
    assert "boom" in job.error
    assert "report" in job.files and "coaching" not in job.files


def test_coaching_is_skipped_when_the_upload_opted_out(tmp_path):
    store = JobStore(tmp_path)
    runner = StubRunner(store)
    job = _decode(store, runner, coach=False)

    assert job.status == "done", "opting out means no pick is asked for either"
    assert not any(argv[0] in BACKENDS for argv in runner.calls)
    assert "coaching" not in job.files
    assert "report" in job.files


def test_queue_drains_in_the_background(tmp_path):
    store = JobStore(tmp_path)
    runner = StubRunner(store)
    job = _job_with_demo(store)

    runner.submit(job.id)
    _wait_for(store, job.id, "awaiting_player")
    assert store.get(job.id).status == "awaiting_player"

    job.player = job.roster[0]["hero"]
    store.save(job)
    runner.submit(job.id, stage="coach")
    _wait_for(store, job.id, "done")
    assert store.get(job.id).status == "done"


def _wait_for(store, job_id, status, timeout=5):
    deadline = time.monotonic() + timeout
    while store.get(job_id).status != status and time.monotonic() < deadline:
        time.sleep(0.02)


class StubChatter(Chatter):
    def __init__(self, store, *, rc=0, out="45 秒那波是 3v1。"):
        super().__init__(store)
        self.calls: list[list[str]] = []
        self.stdins: list[str | None] = []
        self.prompts: list[str] = []
        self.rc = rc
        self.out = out

    def _exec(self, argv, *, timeout, stdin_text=None, cwd=None):
        self.calls.append(argv)
        self.stdins.append(stdin_text)
        self.prompts.append(argv[-1])
        if argv[0] == "claude":
            return subprocess.CompletedProcess(argv, self.rc, self.out, "claude blew up")
        return _fake_model(argv, self.out, self.rc)


def _coached_job(tmp_path):
    store = JobStore(tmp_path / "jobs")
    return store, _decode_and_pick(store, StubRunner(store))


def test_the_review_opens_a_session_that_chat_resumes(tmp_path):
    """The whole point: the model never has to be shown the report twice."""
    store, job = _coached_job(tmp_path)
    assert job.sessions["claude"], "the coaching pass must name its session"

    chatter = StubChatter(store)
    result = chatter.ask(job, "15 分那波呢？")

    assert result["reply"] == "45 秒那波是 3v1。"
    assert chatter.calls[0][:4] == ["claude", "-p", "--resume", job.sessions["claude"]]
    assert chatter.stdins == [None]  # resumed, so no report is re-sent


def test_chat_without_a_review_bootstraps_its_own_session(tmp_path):
    """`--no-coach` jobs are still worth talking to; the first question carries
    the report in and names a session for every question after it."""
    store = JobStore(tmp_path / "jobs")
    job = _decode(store, StubRunner(store), coach=False)
    assert job.sessions == {}

    chatter = StubChatter(store)
    chatter.ask(job, "我對線期哪裡有問題？")

    assert chatter.calls[0][:3] == ["claude", "-p", "--session-id"]
    assert chatter.stdins[0] == "# report"
    assert job.sessions["claude"] == chatter.calls[0][3]

    chatter.ask(job, "那中期呢？")
    assert chatter.calls[1][:4] == ["claude", "-p", "--resume", job.sessions["claude"]]
    assert chatter.stdins[1] is None


def test_chat_history_survives_a_restart(tmp_path):
    store, job = _coached_job(tmp_path)
    StubChatter(store).ask(job, "第一個問題")

    revived = JobStore(tmp_path / "jobs")
    reloaded = revived.get(job.id)
    turns = Chatter(revived).history(reloaded)

    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["text"] == "第一個問題"
    assert reloaded.sessions == job.sessions
    assert reloaded.chat_turns == 2


def test_a_failed_turn_does_not_poison_the_transcript(tmp_path):
    store, job = _coached_job(tmp_path)
    chatter = StubChatter(store, rc=1, out="")

    with pytest.raises(ChatError, match="claude blew up"):
        chatter.ask(job, "會失敗的問題")

    assert chatter.history(job) == []
    assert job.chat_turns == 0


def test_codex_writes_the_review_through_its_own_plumbing(tmp_path):
    """codex puts the answer in the `-o` file and the session id on stdout."""
    store = JobStore(tmp_path)
    runner = StubRunner(store)
    job = _decode_and_pick(store, runner, backend="codex")

    argv = next(a for a in runner.calls if a[0] == "codex")
    assert argv[:4] == ["codex", "exec", "--json", "--sandbox"]
    assert "--skip-git-repo-check" in argv, "a job directory is not a repository"
    assert job.sessions == {"codex": CODEX_THREAD}
    assert (store.dir(job.id) / "coaching.md").read_text(encoding="utf-8").startswith("# 複盤")
    assert runner.coach_stdin == "# report"
    scratch = list(store.dir(job.id).glob(".codex-reply-*"))
    assert scratch == [], "the reply file is a temporary, not an artifact"


def test_a_failed_codex_run_reports_what_it_printed(tmp_path):
    """codex reports failures as stdout events, so an empty stderr is normal."""
    store = JobStore(tmp_path)
    job = _decode_and_pick(store, StubRunner(store, coach_rc=1, coach_out=""), backend="codex")

    assert job.status == "done"
    assert "boom" in job.error
    assert "coaching" not in job.files


def test_switching_engines_hands_the_conversation_over(tmp_path):
    """Quota runs out mid-match; the other engine has to pick the thread up."""
    store, job = _coached_job(tmp_path)
    chatter = StubChatter(store)
    chatter.ask(job, "15 分那波呢？")

    chatter.ask(job, "那第二點呢？", backend="codex")

    opened = chatter.calls[1]
    assert opened[0] == "codex" and "resume" not in opened
    assert chatter.stdins[1] == "# report", "the new engine has never seen the report"
    assert "15 分那波呢？" in chatter.prompts[1], "nor the conversation so far"
    assert job.sessions == {"claude": job.sessions["claude"], "codex": CODEX_THREAD}
    assert job.backend == "codex", "the job stays where it was last answered"
    assert [t.get("backend") for t in chatter.history(job) if t["role"] == "assistant"] == [
        "claude",
        "codex",
    ]

    # Switching back must reuse the session that engine already has.
    chatter.ask(job, "回來問 claude", backend="claude")
    assert chatter.calls[2][:4] == ["claude", "-p", "--resume", job.sessions["claude"]]
    assert chatter.stdins[2] is None


def test_handover_carries_only_the_tail_of_the_conversation(tmp_path):
    turns = [{"role": "user", "text": f"問題 {i}"} for i in range(20)]
    recap = handover(turns)

    assert "問題 19" in recap and "問題 14" in recap
    assert "問題 13" not in recap, "older turns are dropped, not summarised"
    assert recap.startswith("Another engine already handled")
    assert handover([]) == ""


def test_codex_session_id_is_read_off_the_event_stream():
    stdout = (
        '{"type":"thread.started","thread_id":"abc-123"}\n'
        'Reading additional input from stdin...\n'   # not JSON, and not fatal
        '{"type":"turn.completed"}\n'
    )
    assert codex_session(stdout) == "abc-123"
    assert codex_session("") is None
    assert codex_session("not json at all") is None


def test_an_old_job_keeps_the_session_it_was_saved_with(tmp_path):
    """Jobs outlive the demo that made them; a rename must not orphan one."""
    store = JobStore(tmp_path)
    job = store.create(original_name="x.dem", player=None, coach=True, keep_demo=False)
    path = store.dir(job.id) / "job.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["sessions"], data["backend"], data["language"]
    data["session_id"] = "old-uuid"   # what the pre-backend build wrote
    path.write_text(json.dumps(data), encoding="utf-8")

    reloaded = JobStore(tmp_path).get(job.id)
    assert reloaded is not None, "an old job must still load"
    assert reloaded.sessions == {"claude": "old-uuid"}
    assert reloaded.backend == "claude"
    assert reloaded.language == "zh-TW", "pre-localization reviews were Chinese"


def test_chat_refuses_a_job_that_has_no_report(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = store.create(original_name="x.dem", player=None, coach=True, keep_demo=False)

    with pytest.raises(ChatError):
        StubChatter(store).ask(job, "有東西可以聊嗎？")


class _NoRunner(Runner):
    """Accept jobs without doing anything, so API tests stay hermetic."""

    def _loop(self):
        while True:
            self._queue.get()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from deadlock_coach import server

    monkeypatch.setattr(server, "Runner", _NoRunner)
    return TestClient(server.create_app(data_dir=tmp_path))


def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Deadlock Replay Coach" in response.text


def test_index_uses_a_sidebar_and_one_active_replay_workspace(client):
    page = client.get("/").text

    assert '<html lang="en">' in page
    assert 'aria-label="Analyzed replays"' in page
    assert 'id="language"' in page
    assert 'id="detail-view"' in page
    assert "function selectJob(id)" in page
    assert "jobs.map(j =>" in page
    assert "function drawViewer(id)" in page
    assert "戰術地圖與當時狀態" in page
    assert "function toggleViewerHeroVisibility(id, heroId, visible)" in page
    assert "function toggleViewerLayer(id, layer)" in page
    assert "地圖人物" in page
    assert "這次最想分析什麼？" in page
    assert "function jumpToClock(id, clock)" in page
    assert 'class="review-workspace"' in page
    assert 'class="review-visual"' in page
    assert 'class="review-reading"' in page
    assert "function setReviewPane(id, pane)" in page
    assert "visualPane.scrollTo" in page
    assert "scrollIntoView" not in page


def test_index_puts_the_numeric_gap_summary_before_long_coaching(client):
    page = client.get("/").text

    assert "跟最高段位差在哪裡" in page
    assert "你的命中率" in page
    assert "最高段位最大實測差距" in page
    assert "這場最優先處理的" in page
    assert "查看所有比率、最高段位對照與判定依據" in page
    assert "查看教練解讀" in page
    assert '<details class="score-details">' in page
    assert '<details class="coach-details">' in page


def test_index_is_never_cached(client):
    # The page inlines its own JS, so a cached copy means a device keeps running
    # an old frontend against a restarted server and appears broken.
    assert client.get("/").headers["cache-control"] == "no-store"


def test_upload_rejects_anything_that_is_not_a_dem(client):
    response = client.post("/api/upload", files={"demo": ("notes.txt", b"hello")})
    assert response.status_code == 400
    assert client.get("/api/jobs").json()["jobs"] == []


def test_upload_rejects_an_empty_file_and_leaves_no_job_behind(client):
    response = client.post("/api/upload", files={"demo": ("empty.dem", b"")})
    assert response.status_code == 400
    assert client.get("/api/jobs").json()["jobs"] == []


def test_upload_size_limit_removes_the_partial_job(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from deadlock_coach import server

    monkeypatch.setattr(server, "Runner", _NoRunner)
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 4)
    with TestClient(server.create_app(data_dir=tmp_path)) as local:
        response = local.post("/api/upload", files={"demo": ("large.dem", b"12345")})
        assert response.status_code == 413
        assert local.get("/api/jobs").json()["jobs"] == []
    assert list((tmp_path / "jobs").iterdir()) == []


def test_upload_queues_a_job_with_no_perspective_assumed(client):
    response = client.post(
        "/api/upload",
        files={"demo": ("12345678.dem", b"PBDEMS2" * 100)},
        data={"coach": "true", "keep_demo": "false", "player": "Yamato"},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "queued"
    assert job["size_bytes"] == 700
    assert job["language"] == "en"
    # Even if a client sends one, upload is not where a perspective is decided.
    assert job["player"] is None


def test_config_lists_both_engines_for_the_page(client):
    config = client.get("/api/config").json()

    assert [b["name"] for b in config["backends"]] == ["claude", "codex"]
    assert all("available" in b and "label" in b for b in config["backends"])
    assert config["default_backend"] in BACKENDS


def test_upload_records_the_engine_the_page_asked_for(client):
    job_id = client.post(
        "/api/upload",
        files={"demo": ("x.dem", b"PBDEMS2")},
        data={"backend": "codex"},
    ).json()["id"]
    assert client.get(f"/api/jobs/{job_id}").json()["backend"] == "codex"

    # An engine nobody implements is a bug in the page, not a job to run.
    rejected = client.post(
        "/api/upload",
        files={"demo": ("y.dem", b"PBDEMS2")},
        data={"backend": "gpt-9"},
    )
    assert rejected.status_code == 400

    # Omitting it is normal — old clients, and the form before /api/config lands.
    plain = client.post("/api/upload", files={"demo": ("z.dem", b"PBDEMS2")}).json()["id"]
    assert client.get(f"/api/jobs/{plain}").json()["backend"] in BACKENDS


def test_upload_records_and_validates_the_report_language(client):
    job_id = client.post(
        "/api/upload",
        files={"demo": ("zh.dem", b"PBDEMS2")},
        data={"language": "zh-TW"},
    ).json()["id"]
    assert client.get(f"/api/jobs/{job_id}").json()["language"] == "zh-TW"

    rejected = client.post(
        "/api/upload",
        files={"demo": ("bad.dem", b"PBDEMS2")},
        data={"language": "fr"},
    )
    assert rejected.status_code == 400


def test_upload_records_the_requested_analysis_focus(client):
    focus = "  請看我的地圖站位和出裝  "
    job_id = client.post(
        "/api/upload",
        files={"demo": ("focus.dem", b"PBDEMS2")},
        data={"analysis_request": focus},
    ).json()["id"]

    assert client.get(f"/api/jobs/{job_id}").json()["analysis_request"] == focus.strip()


def test_the_hero_pick_can_change_the_engine_before_it_costs_anything(tmp_path):
    from fastapi.testclient import TestClient

    from deadlock_coach import server

    store = JobStore(tmp_path / "jobs")
    job = _decode(store, StubRunner(store))
    assert job.backend == "claude"

    with TestClient(server.create_app(data_dir=tmp_path)) as client:
        picked = client.post(
            f"/api/jobs/{job.id}/player",
            json={
                "hero_id": 12,
                "backend": "codex",
                "language": "zh-TW",
                "analysis_request": "請看 12:44 的地圖站位",
            },
        )
        assert picked.status_code == 200
        assert picked.json()["backend"] == "codex"
        assert client.get(f"/api/jobs/{job.id}").json()["backend"] == "codex"
        assert client.get(f"/api/jobs/{job.id}").json()["language"] == "zh-TW"
        assert client.get(f"/api/jobs/{job.id}").json()["analysis_request"] == "請看 12:44 的地圖站位"


def test_picking_a_hero_is_rejected_unless_the_job_is_waiting(client):
    job_id = client.post("/api/upload", files={"demo": ("x.dem", b"PBDEMS2")}).json()["id"]

    # still queued, so there is no roster to pick from yet
    r = client.post(f"/api/jobs/{job_id}/player", json={"hero_id": 12})
    assert r.status_code == 409
    assert "queued" in r.json()["detail"]
    assert client.post("/api/jobs/nope/player", json={"hero_id": 12}).status_code == 404


def test_picking_from_the_roster_sets_the_perspective(tmp_path):
    from fastapi.testclient import TestClient

    from deadlock_coach import server

    store = JobStore(tmp_path / "jobs")
    job = _decode(store, StubRunner(store))
    assert job.status == "awaiting_player"

    with TestClient(server.create_app(data_dir=tmp_path)) as client:
        assert client.post(f"/api/jobs/{job.id}/player", json={"hero_id": 999}).status_code == 400

        picked = client.post(f"/api/jobs/{job.id}/player", json={"hero_id": 12})
        assert picked.status_code == 200
        expected = next(r["hero"] for r in job.roster if r["hero_id"] == 12)
        assert picked.json()["player"] == expected
        assert client.get(f"/api/jobs/{job.id}").json()["status"] in ("queued", "coaching", "done")


def test_declining_to_pick_reviews_the_whole_match(tmp_path):
    from fastapi.testclient import TestClient

    from deadlock_coach import server

    store = JobStore(tmp_path / "jobs")
    job = _decode(store, StubRunner(store))

    with TestClient(server.create_app(data_dir=tmp_path)) as client:
        r = client.post(f"/api/jobs/{job.id}/player", json={"hero_id": None})
        assert r.status_code == 200
        assert r.json()["player"] is None


def test_chat_endpoint_validates_before_spawning_anything(client):
    job_id = client.post("/api/upload", files={"demo": ("x.dem", b"PBDEMS2")}).json()["id"]

    assert client.post(f"/api/jobs/{job_id}/chat", json={"message": "  "}).status_code == 400
    assert client.post("/api/jobs/nope/chat", json={"message": "hi"}).status_code == 404
    # queued job, so no report yet
    assert client.post(f"/api/jobs/{job_id}/chat", json={"message": "hi"}).status_code == 502
    assert client.get(f"/api/jobs/{job_id}/chat").json() == {"messages": [], "can_chat": False}


def test_missing_job_and_missing_artifact_are_404_not_500(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.delete("/api/jobs/nope").status_code == 404

    job_id = client.post("/api/upload", files={"demo": ("x.dem", b"PBDEMS2")}).json()["id"]
    assert client.get(f"/api/jobs/{job_id}/file/report").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/text/coaching").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/viewer").status_code == 404


def test_visual_replay_is_served_as_immutable_json(tmp_path):
    from fastapi.testclient import TestClient

    from deadlock_coach import server

    store = JobStore(tmp_path / "jobs")
    job = store.create(original_name="map.dem", player=None, coach=False, keep_demo=False)
    viewer = store.dir(job.id) / "map.viewer.json"
    viewer.write_text('{"schema_version":1}', encoding="utf-8")
    job.files["viewer"] = viewer.name
    store.save(job)

    with TestClient(server.create_app(data_dir=tmp_path)) as local:
        response = local.get(f"/api/jobs/{job.id}/viewer")
    assert response.status_code == 200
    assert response.json()["schema_version"] == 2
    assert len(response.json()["map"]["landmarks"]) == 40
    assert "immutable" in response.headers["cache-control"]


def test_downloads_are_named_after_the_match_without_doubling_the_stem(tmp_path):
    """`coaching.md` must be disambiguated; the pipeline's files already are."""
    from fastapi.testclient import TestClient

    from deadlock_coach import server

    store = JobStore(tmp_path / "jobs")
    job = _decode_and_pick(store, StubRunner(store))

    app = server.create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        names = {
            kind: client.get(f"/api/jobs/{job.id}/file/{kind}").headers["content-disposition"]
            for kind in ("report", "json", "viewer", "coaching")
        }
    assert 'filename="12345678.report.md"' in names["report"]
    assert 'filename="12345678.match.json"' in names["json"]
    assert 'filename="12345678.viewer.json"' in names["viewer"]
    assert 'filename="12345678.coaching.md"' in names["coaching"]


def test_delete_removes_the_job_and_its_files(client, tmp_path):
    job_id = client.post("/api/upload", files={"demo": ("x.dem", b"PBDEMS2")}).json()["id"]
    assert (tmp_path / "jobs" / job_id).exists()

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert not (tmp_path / "jobs" / job_id).exists()
    assert client.get("/api/jobs").json()["jobs"] == []


def test_job_serialises_every_field_the_page_reads(tmp_path):
    """The page indexes these directly; a rename here is a silent blank in the UI."""
    from dataclasses import asdict

    job = Job(id="x", stem="x", original_name="x.dem", created="now")
    keys = asdict(job).keys()
    for field in ("status", "player", "backend", "language", "analysis_request", "error", "log", "files",
                  "parse_seconds", "coach_seconds"):
        assert field in keys
