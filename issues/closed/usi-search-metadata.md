# USI Search Metadata In Game Logs

Status: closed.

## Issue

`ShogiGameRecord` previously stored players, moves, winner, and end reason, but
the USI wrapper discarded search metadata emitted through `info ...` lines.

This is weak for using USI engines as teacher data sources. Game logs are
source records for later learning-data conversion, so information that may
affect source priority, filtering, weighting, or target construction should not
be dropped at log time. Engine strength, teacher confidence, candidate
distribution, and value-like signals often live in per-move search metadata,
not only in the final `bestmove`.

## Implemented

- `usi` / `usiok`
- `isready` / `readyok`
- `position startpos ...`
- `position sfen ...`
- configurable `go ...`
- `bestmove`
- `bestmove resign`
- ponder move from `bestmove ... ponder ...`
- raw `info ...` lines emitted before each `bestmove`
- player settings recorded by this repository, such as command, go command,
  checkpoint, policy, and MCTS simulations

## Remaining Outside This Issue

- `id name` and `id author`
- `option name ...`
- `bestmove win`

## Candidate Direction

Keep the structure simple, but prefer preserving USI search metadata as emitted
over minimizing log size. The log should remain a source record, not a
training-ready example. Do not collapse multiple `info ...` lines into a single
final or best value at log time; later conversion code can decide which fields
or lines to use.

The first useful scope is:

- raw `info ...` lines emitted before each `bestmove`
- parsed views for common fields when useful, such as score, depth, seldepth,
  nodes, nps, time, pv, and multipv
- bestmove
- ponder move, if present

Do not try to model every USI option up front. The runtime repository should
capture raw enough metadata for later learning-data conversion, while
`../intelligence-representation` decides how to convert that metadata into
policy targets, value targets, source priority, filtering, or weights.

## Acceptance Criteria

This issue is closed because:

- external USI process calls preserve emitted `info ...` lines alongside
  `bestmove`,
- game logs preserve per-move search metadata when available,
- tests cover `score`, `depth`, `nodes`, `pv`, and `multipv` parsing or
  preservation, and
- the resulting JSONL remains readable by `intelligence-representation` as a
  raw source record.
