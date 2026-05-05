# USI Search Metadata In Game Logs

Status: open.

## Issue

`ShogiGameRecord` currently stores players, moves, winner, and end reason, but
the USI wrapper discards search metadata emitted through `info ...` lines.

This is weak for using USI engines as teacher data sources. Game logs are
source records for later learning-data conversion, so information that may
affect source priority, filtering, weighting, or target construction should not
be dropped at log time. Engine strength, teacher confidence, candidate
distribution, and value-like signals often live in per-move search metadata,
not only in the final `bestmove`.

## Currently Used

- `usi` / `usiok`
- `isready` / `readyok`
- `position startpos ...`
- `position sfen ...`
- configurable `go ...`
- `bestmove`
- `bestmove resign`
- player settings recorded by this repository, such as command, go command,
  checkpoint, policy, and MCTS simulations

## Currently Ignored Or Dropped

- `id name` and `id author`
- `option name ...`
- `info score cp ...`
- `info score mate ...`
- `info pv ...`
- `info depth ...`
- `info seldepth ...`
- `info nodes ...`
- `info nps ...`
- `info time ...`
- `info multipv ...`
- ponder move from `bestmove ... ponder ...`
- `bestmove win`

## Candidate Direction

Keep the structure simple, but prefer preserving raw-ish USI search metadata
over minimizing log size. The log should remain a source record, not a
training-ready example, but it should retain enough per-move metadata for later
conversion decisions.

The first useful scope is:

- bestmove
- ponder move, if present
- final or best available score
- final or best available depth
- final or best available nodes
- principal variation
- MultiPV candidate lines when enabled

Do not try to model every USI option up front. The runtime repository should
capture raw enough metadata for later learning-data conversion, while
`../intelligence-representation` decides how to convert that metadata into
policy targets, value targets, source priority, filtering, or weights.

## Acceptance Criteria

This issue can close when:

- external USI process calls return selected `info ...` metadata alongside
  `bestmove` when the engine emits it,
- game logs preserve per-move search metadata when available,
- tests cover `score`, `depth`, `nodes`, `pv`, and `multipv` parsing or
  preservation, and
- the resulting JSONL remains readable by `intelligence-representation` as a
  raw source record.
