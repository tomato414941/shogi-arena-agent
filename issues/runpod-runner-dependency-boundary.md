# RunPod Runner Dependency Boundary

Status: open. Priority: medium.

## Issue

`scripts/runpod_evaluate_checkpoint_vs_yaneuraou.sh` provides the repository
entrypoint for RunPod arena evaluation, but it currently delegates pod
lifecycle management to:

```text
../intelligence-representation/scripts/runpod/run_job.py
```

This keeps the RunPod API, SSH, rsync, and pod deletion implementation in one
place, but it makes this repository's RunPod arena evaluation depend on a
sibling repository tool.

## Current Position

This is acceptable as a short-term operating shortcut.

It does not reverse the model/runtime code dependency direction:

```text
shogi-arena-agent -> intelligence-representation
```

However, it is weaker as an operations boundary because `shogi-arena-agent`
cannot run its RunPod arena evaluation entrypoint standalone unless
`intelligence-representation` is checked out beside it.

## Risks

- sibling checkout layout becomes part of the implicit contract,
- changes to the `intelligence-representation` RunPod runner can break arena
  evaluation,
- RunPod job execution responsibility remains partly owned by the model
  research repository,
- moving beyond checkpoint-vs-YaneuraOu evaluation may make this dependency
  harder to reason about.

## Candidate Directions

Keep the current wrapper while RunPod arena evaluation is still light.

Revisit when RunPod arena evaluation becomes a regular workflow. Plausible
directions:

- move the generic RunPod job runner to a shared operations repository,
- make `shogi-arena-agent` own its own runner if arena runtime operations
  diverge from training jobs,
- keep one runner but document the sibling-repository contract explicitly if it
  remains intentional.

Avoid copying the whole RunPod runner into this repository until there is a
clear need, because duplicated pod lifecycle code is likely to drift.

## Acceptance Criteria

This issue can close when one of the following is true:

- RunPod job lifecycle code has a clear shared home outside either project,
- `shogi-arena-agent` owns a minimal runner appropriate for arena evaluation,
  or
- the sibling `intelligence-representation` runner dependency is intentionally
  documented as the supported operations contract.
