# USI Process Stderr Handling

Status: closed.

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

## Implemented

- `UsiProcess` sends engine stderr to `subprocess.DEVNULL` by default.
- USI stdout remains captured for protocol responses and `info ...` search
  metadata.
- Tests cover an engine that writes to stderr before returning `bestmove`.

## Acceptance Criteria

This issue is closed because:

- long-running USI calls cannot block on an undrained stderr pipe, and
- the default path preserves stdout USI metadata while intentionally discarding
  stderr diagnostics.
