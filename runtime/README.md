# MacMimic OpenSprite runtime

This directory contains the MacMimic 100k runtime deployed on Jetson 253. See the [sim-to-real guide](../sim2real/README.md) for the current build, controls, qualifications and test evidence.

## Upstream snapshot history

Snapshot of `BeijingDynamics/open_sprite_runtime`, branch `codex/supermimic-v5`,
commit `5f8225ce6258cb38db174ee18c55162fa34d96a6`.
This project additionally maintains the bounded C++ zero-gain probe and persistent
SPI0 performance policy verified on 2026-11-06; see the parent SPI power-management guide.
The snapshot includes native 10-frame/993-input replay, receive-only IPC, motor maps,
strict closed-loop simulation, tests and their dependencies. No model is hardware qualified.

Use the parent project's [CPU and closed-loop instructions](https://github.com/TonyInvent/OpenSprite-SuperMimic/blob/fe4e74f96117debbd66e5511af256418e1edcd23/docs/RUNTIME.md).
The recommended commands build only file-replay and test targets, and never enable motors.
Other historical commissioning tools are retained as source dependencies; they are not
the entry point for this model. No deployed controller is replaced by installing this package.
