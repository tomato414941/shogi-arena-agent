# USI Process Stderr Handling

Status: open.

## Issue

`UsiProcess` starts external engines with `stderr=subprocess.PIPE`, but the
stderr stream is not drained while the engine is running.

This can hang a long game-generation or evaluation run if an engine writes
enough data to stderr to fill the pipe. The failure mode is operationally
confusing because the caller may only see a USI response timeout even though
the engine process is blocked on stderr output.

## Current State

`UsiProcess.start()` pipes stdin, stdout, and stderr. A background thread drains
stdout into a queue, but stderr is only read after an unexpected exit.

## Candidate Direction

Keep this simple. If stderr is not needed during normal operation, send it to
`subprocess.DEVNULL`. If debug visibility is needed later, add an explicit
debug/log option that drains stderr in a background thread.

Do not make stderr capture part of the default game record. USI search metadata
belongs in stdout `info ...` lines, not stderr.

## Acceptance Criteria

- Long-running USI calls cannot block because stderr is not drained.
- Unexpected engine exits still report enough information to debug basic
  startup failures, or the limitation is explicit.
- Tests cover an engine that writes to stderr before returning `bestmove`.
