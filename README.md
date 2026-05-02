# shogi-arena-agent

Runtime integration project for putting a shogi model into external arenas.

This repository is intentionally separate from `intelligence-representation`.
The model research repository should stay focused on training, checkpoints, and
portable inference boundaries. This repository owns shogi-specific runtime
concerns such as USI, local engine matches, Lishogi Bot integration, and future
arena deployment.

## Scope

In scope:

- USI engine wrapper
- local shogi arena runs
- Lishogi Bot bridge
- optional Floodgate or tournament integration
- loading exported model checkpoints from `intelligence-representation`

Out of scope:

- model architecture research
- training recipes
- general multimodal representation design
- changes that make `intelligence-representation` depend on this repository

## Dependency Direction

```text
shogi-arena-agent
  may depend on intelligence-representation

intelligence-representation
  must not depend on shogi-arena-agent
```

## Initial Milestones

1. Define the minimal inference boundary for a shogi position.
2. Load a local placeholder policy and return a legal move.
3. Wrap that policy behind a USI-compatible process.
4. Run local games against a baseline engine.
5. Evaluate Lishogi Bot integration.

## Local USI Smoke

```sh
printf 'usi\nisready\nposition startpos\ngo btime 0 wtime 0\nquit\n' \
  | uv run python -m shogi_arena_agent
```

Example response:

```text
id name shogi-arena-agent
id author intrep
usiok
readyok
bestmove <legal-usi-move>
```

The default policy is intentionally simple: it reconstructs the current board
with `python-shogi` and returns one deterministic legal move.

## Local Match Smoke

```sh
uv run python - <<'PY'
from shogi_arena_agent.local_match import play_local_match

print(play_local_match(max_plies=8))
PY
```

This runs two local `UsiEngine` instances against each other and verifies each
returned `bestmove` before applying it to the board.

To exercise the real process boundary:

```sh
uv run python - <<'PY'
from shogi_arena_agent.local_match import play_local_match
from shogi_arena_agent.usi_process import UsiProcess

with UsiProcess() as black, UsiProcess() as white:
    print(play_local_match(black=black, white=white, max_plies=8))
PY
```
