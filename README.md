# deadlock-replay-coach

Turn a Deadlock `.dem` replay into an AI-readable coaching report, structured
match data, and a synchronized tactical replay viewer.

The decoder reduces millions of per-tick values into evidence an AI coach can
reason about: fight outcomes, objective conversion, economy, isolation deaths,
movement, mechanics, and player-specific missed opportunities.

English is the default for the web interface, new coaching reports, follow-up
answers, prompts, CLI messages, and bundled map labels. Traditional Chinese is
available from the language picker and is only used when selected.

## Quick start

### 1. Prerequisites

- macOS, Linux, or Windows with WSL
- Python 3.11–3.14 (a current limitation of
  [boon](https://github.com/pnxenopoulos/boon))
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A Deadlock `.dem` replay
- Optional: a logged-in Codex CLI or Claude Code installation for AI coaching

After cloning this repository, install the server dependencies:

```bash
cd deadlock-replay-coach
uv sync --extra server
```

### 2. Connect your own AI subscription

The app does not need an API key. It invokes a supported CLI that is already
logged in on your computer, so credentials stay in that CLI's own secure
storage and are never written to this repository or `server-data/`.

Install and sign in to at least one of the following engines.

#### Option A: Codex with a ChatGPT plan

[Codex is included with ChatGPT plans](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan),
with usage limits depending on the plan.

```bash
npm install -g @openai/codex
codex login
codex login status
```

Choose **Sign in with ChatGPT** in the browser flow. No API key needs to be
copied into this project. See the
[official Codex repository](https://github.com/openai/codex) for alternative
installation methods.

#### Option B: Claude Code with a Claude Pro or Max plan

[Claude Pro and Max include Claude Code](https://support.anthropic.com/en/articles/11145838-using-claude-code-with-your-pro-or-max-plan).

```bash
npm install -g @anthropic-ai/claude-code
claude
```

On first launch, select the Claude.ai account associated with your subscription.
If Claude Code is currently using Anthropic Console pay-as-you-go billing, run
`/login` inside Claude Code and select the subscription account. See Anthropic's
[official setup guide](https://docs.anthropic.com/en/docs/claude-code/getting-started)
for other installation methods.

You may install both engines. The web UI lets you switch before generating a
review and again for every follow-up question. Usage is charged against the
selected provider's plan and limits.

### 3. Start the web app

```bash
uv run deadlock-coach-server
```

The terminal prints two URLs:

- `http://localhost:8000` for this computer
- A LAN URL such as `http://192.168.x.x:8000` for another device on the same Wi-Fi

Open either URL, then:

1. Drop in a `.dem` replay.
2. Optionally describe what you want the coach to focus on.
3. Select Codex or Claude and click **Upload & analyze**.
4. After decoding, choose the hero you played.
5. Read the English review, inspect the tactical map, or ask follow-up questions.

The first installed engine is selected automatically. To choose the initial
default explicitly:

```bash
uv run deadlock-coach-server --backend codex
uv run deadlock-coach-server --backend claude
```

If neither engine is installed, the server starts in decode-only mode. You can
also request that mode directly:

```bash
uv run deadlock-coach-server --no-coach
```

## CLI usage

AI coaching is optional. To decode a replay without running the web server:

```bash
uv sync
uv run deadlock-coach /path/to/match.dem -o out/
```

This writes:

```text
out/match.report.md   AI-readable Markdown report
out/match.match.json  Structured match evidence
out/match.summary.json Per-player ranked summaries
out/match.viewer.json Tactical replay data
```

Useful options:

```bash
# Focus the report on one hero or player name
uv run deadlock-coach match.dem -o out/ --player Yamato

# Save the underlying Polars frames as parquet
uv run deadlock-coach match.dem -o out/ --dump-frames

# Never call the Deadlock APIs; use bundled data and replay metadata
uv run deadlock-coach match.dem -o out/ --offline

# Change positional sampling or the report timeline cap
uv run deadlock-coach match.dem -o out/ --sample-seconds 10 --max-events 300
```

Python API:

```python
from deadlock_coach import load_demo, render_report

match = load_demo("match.dem")
print(render_report(match))
```

## Language behavior

- A fresh browser session starts in English.
- New jobs default to English at both the browser and server API layers.
- The selected language is saved with each job and is reused for follow-up
  questions.
- Choosing Traditional Chinese in the upper-right language picker changes both
  the interface and newly generated coaching output.
- English prompts live in `deadlock_coach/coach_prompt_en.md` and
  `deadlock_coach/chat_prompt_en.md`.
- Traditional Chinese prompts live in `deadlock_coach/coach_prompt.md` and
  `deadlock_coach/chat_prompt.md`.

## What the web app does

For each upload, the server:

1. Runs the same decoder as the CLI in an isolated subprocess.
2. Waits for you to identify your hero instead of guessing your identity.
3. Sends a player-focused report to the selected, locally authenticated AI CLI.
4. Shows the coaching report next to a synchronized tactical map with positions,
   objectives, neutral camps, ziplines, inventory, and ability state.
5. Keeps one resumable conversation per engine, so switching providers preserves
   recent context.
6. Lets you download the Markdown and JSON artifacts.

The uploaded `.dem` is deleted after parsing unless **Keep .dem** is selected.
Reports and JSON artifacts remain under `server-data/jobs/<id>/` and survive a
server restart. Decode jobs run one at a time to avoid excessive memory use;
chat requests do not use that queue. The upload limit is 2 GiB.

## Privacy, network access, and security

- No API keys, AI credentials, replays, generated reports, chat histories, or
  local cache files are committed. `.gitignore` excludes `.dem`, `out/`,
  `analysis/`, `server-data/`, local virtual environments, caches, and builds.
- Replay parsing happens locally. When coaching is enabled, the focused report
  and follow-up messages are sent to the selected AI provider through its CLI.
- Unless `--offline` is used, the decoder may contact Deadlock data APIs for
  current game constants, assets, and fallback post-match statistics.
- The web server has no authentication and binds to `0.0.0.0` by default. Use it
  only on a trusted LAN. Do not expose it directly to the public internet; use
  a private network such as Tailscale and add authentication if remote access is
  required.
- Player names contained in a replay naturally appear in that replay's generated
  local report. Review generated artifacts before sharing them publicly.

## Report contents

- **Bottom line** — ranked, player-specific conclusions with denominators,
  same-hero rank baselines, evidence types, and links to detailed sections.
- **Match and roster** — map, winner, lineups, and final statistics.
- **Phases** — lane, mid, and late-game boundaries based on actual match events,
  plus decisive-fight records for each phase.
- **Per-player phase statistics** — phase deltas for KDA, souls, hero damage,
  last hits, and objective damage.
- **Soul economy** — net-worth curves, maximum leads, and major swings.
- **Win conditions and advantage ledger** — farming, denies, jungle control,
  invasions, kills, structures, Mid Boss/Rift, positioning, and death downtime.
- **Teamfights** — participants, result, damage, and whether a won fight converted
  into a permanent resource within 45 seconds, with creep-wave context.
- **Mechanics and movement** — movement tiers, accuracy, headshot rate, bullets
  directed at heroes, and last-hit efficiency.
- **Same-hero population comparison** — rate-based comparisons with the player's
  rank and the highest rank with enough samples. Style-only metrics are not
  scored as better or worse.
- **Kill patterns** — picks, isolation deaths, and whether teammates could have
  arrived under the movement model.
- **Player-perspective opportunities** — estimated kill pressure, cross-lane
  rotations, macro windows, and invade/scout opportunities using only information
  the player could reasonably have had.
- **Event timeline** — objective events plus an even sample across the full match.

Unknown and estimated values are labeled explicitly. The report does not replace
missing evidence with zero and does not present detector output as measured fact.

## Design notes

A typical match contains roughly 60 ticks/second × 30 minutes × 12 players, or
about 1.3 million rows of player state before other entities are counted. Raw
coordinates are both too large for an LLM context window and poor material for
tactical reasoning. This project reduces them into three layers:

| Layer | Content | Consumer |
|---|---|---|
| Structured | Polars frames and optional parquet dumps | Code and tool calls |
| Event stream | Natural-language events with match clocks | LLM |
| Tactical reads | Fights, conversion, economy, picks, and opportunities | LLM |

Important implementation choices:

- Missing samples remain `unknown` instead of becoming false zeroes.
- Positions are sampled once per second by default, while every kill tick is
  retained exactly.
- Net worth is `souls + spent_souls`; looking only at unspent souls misranks
  players immediately after purchases.
- KDA comes from the kill feed so incomplete final scoreboard samples do not
  silently change it.
- Travel time is preferred over raw distance. The movement model is a lower
  bound based on straight-line, obstacle-free, full-stamina travel.
- Game constants and visual assets refresh from the Deadlock APIs and fall back
  to bundled snapshots. `--offline` disables all such requests.

## Architecture

```text
gamedata.py       Deadlock data APIs, movement constants, assets, rank baselines
physics.py        Distance/time conversion and movement-mechanism inference
replay_stats.py   PostMatchDetails counters embedded in the replay
skillstats.py     Replay statistics with API fallback
benchmark.py      Same-hero, same-rank population comparisons
source.py         .dem to MatchData; the only module that imports boon
viewer.py         Positions, inventory, abilities, map assets, and viewer JSON
events.py         Narrative event timeline
tactics.py        Phases, fights, picks, economy, and coaching notes
opportunities.py  Player-perspective tactical opportunity detectors
advantage.py      Resource ledger, map control, and win-condition windows
summary.py        Ranked per-player summaries
render.py         Markdown and JSON renderers
server.py         Upload queue, AI CLI integration, resumable chat, and web API
```

The analysis modules consume `MatchData`, not boon directly, which keeps the
test suite fast and allows synthetic fixtures.

## Tests

```bash
uv run pytest
```

The tests use synthetic data and scripted fake AI CLI responses. They do not
need a real replay, make AI requests, or consume subscription usage. An autouse
fixture prevents network access and forces bundled constants.

## Known limitations

- The tactical map is a 2D replay reconstruction, not game footage. Dead heroes
  stop emitting position rows, so stale markers are shown as last-known positions.
- Map collision geometry is not reconstructed from the VPK. Wall occlusion is
  therefore not exact, and occlusion-dependent conclusions cannot receive high
  confidence on their own.
- Deadlock changes frequently. Unknown objective types are preserved rather than
  causing report generation to fail.
- The decoder has been calibrated against a limited set of real replay builds;
  additional replay samples and bug reports are welcome.

## License

[MIT](LICENSE)
