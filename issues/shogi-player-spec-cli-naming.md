# Shogi Player Spec / CLI Naming

Status: open. Priority: low.

## Issue

The runtime code now uses player spec variants:

- `CheckpointPolicyPlayerSpec`
- `ExternalEnginePlayerSpec`
- `DeterministicLegalPlayerSpec`

The CLI and actor provenance still use legacy kind values:

- `checkpoint`
- `usi_engine`
- `deterministic_legal`

This works, but the names mix different axes. `usi_engine` names the protocol
used by the external engine, while `ExternalEnginePlayerSpec` names the player
construction source. A checkpoint-backed player can also be exposed through USI,
so USI should not be treated as the general player category outside the build
boundary.

## Current Position

Keep the current CLI values for now. The recent refactor already moved `kind`
interpretation to the CLI/build boundary, and runner code now receives
`PlayerSpec` variants.

## Candidate Direction

Decide whether future CLI/provenance names should align with construction
source:

- checkpoint policy
- external engine
- deterministic legal policy

If renamed, keep protocol details such as USI under the external-engine spec
settings rather than making the protocol the top-level player category.

## Close Condition

- The project either keeps the current CLI/provenance names intentionally, or
- renames the CLI/provenance categories so the player category axis is
  consistent with `PlayerSpec`.
