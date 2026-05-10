# RunPod Runner Dependency Boundary

Status: closed.

## Issue

`scripts/runpod_evaluate_checkpoint_vs_yaneuraou.sh` originally provided the
repository entrypoint for RunPod arena evaluation, but delegated pod lifecycle
management to:

```text
../intelligence-representation/scripts/runpod/run_job.py
```

That kept the RunPod API, SSH, rsync, and pod deletion implementation in one
place, but made this repository's RunPod arena evaluation depend on a sibling
model-research repository tool.

## Current Position

This was acceptable as a short-term operating shortcut.

It does not reverse the model/runtime code dependency direction:

```text
shogi-arena-agent -> intelligence-representation
```

However, it was weaker as an operations boundary because `shogi-arena-agent`
could not run its RunPod arena evaluation entrypoint unless
`intelligence-representation` was checked out beside it.

## Implemented

Generic RunPod pod lifecycle code now lives in:

```text
../runpod-job-runner/scripts/run_job.py
```

`scripts/runpod_evaluate_checkpoint_vs_yaneuraou.sh` uses that shared runner by
default through `RUNPOD_RUNNER_ROOT` / `RUNPOD_JOB`. The arena evaluation
wrapper stays in this repository, and the model checkpoint input still comes
from `intelligence-representation`.

## Original Risks

- sibling `intelligence-representation` checkout layout becomes part of the
  implicit contract,
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

This issue is closed because:

- RunPod job lifecycle code has a clear shared home outside either project,
- `shogi-arena-agent` no longer depends on the
  `intelligence-representation` runner implementation, and
- project-specific arena evaluation logic remains in `shogi-arena-agent`.
