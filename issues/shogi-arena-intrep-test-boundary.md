# Shogi Arena / Intrep Test Boundary

Status: open. Priority: medium.

## Issue

`shogi-arena-agent` test discovery currently imports `intrep` internals from at
least one test module. In the current local environment, full unittest discovery
fails before running that test:

```text
ImportError: cannot import name 'ShogiTransitionRecord' from
intrep.worlds.shogi.game_record
```

This is not caused by the player-spec runner refactor. It exposes a weaker
boundary between the arena runtime repository and the model/training repository.

## Why This Matters

`shogi-arena-agent` should be testable as the arena runtime source of truth
without depending on private or unstable `intelligence-representation` types.

The arena may consume artifacts produced by `intelligence-representation`, but
its own tests should avoid importing training-side implementation details unless
the dependency is explicit and stable.

## Candidate Direction

- Keep shared record schemas in one stable package boundary, or
- make the arena test use arena-owned game-record types only, or
- move the integration test to the repository that owns both sides of the
  contract.

Do not paper over this with ad hoc `PYTHONPATH` assumptions.

## Close Condition

- `uv run python -m unittest discover tests` succeeds in `shogi-arena-agent`, or
- the remaining cross-repository test is explicitly isolated as an integration
  test with documented dependency setup.
