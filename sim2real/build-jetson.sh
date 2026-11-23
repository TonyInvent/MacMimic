#!/usr/bin/env bash
# Build the native runtime and verify recorded inference without opening devices.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
runtime="$here/../runtime"
policy="$here/../policies/opensprite-velocity-100k"
work="$here/build/jetson253"
python=${SPRITE_PYTHON:-/home/tony/open_sprite_runtime/.venv/bin/python}
include=${SPRITE_ORT_INCLUDE:-/home/tony/open_sprite_runtime/build/policy_native_stage2/include}
lib=${SPRITE_ORT_LIBRARY:-/home/tony/open_sprite_runtime/.venv/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2}
test -f "$include/onnxruntime_cxx_api.h"
test -f "$lib"
export PYTHONPATH="$runtime/src"
"$python" -m open_sprite_runtime.supermimic_run002 verify --bundle "$policy"
cmake -S "$runtime/native" -B "$work/build" -DCMAKE_BUILD_TYPE=Release \
  -DSPRITE_ORT_INCLUDE="$include" -DSPRITE_ORT_LIBRARY="$lib"
cmake --build "$work/build" -j3
ln -sfn "$lib" "$work/build/libonnxruntime.so.1"
export LD_LIBRARY_PATH="$work/build"
(cd "$work/build" && ctest --output-on-failure)
cmake -S "$runtime/native/icm45686" -B "$work/icm" -DCMAKE_BUILD_TYPE=Release
cmake --build "$work/icm" -j3
(cd "$work/icm" && ctest --output-on-failure)
"$work/build/sprite_supermimic_replay" "$policy/policy.onnx" "$here/evidence/xbox803-input.bin" \
  "$work/replay.bin" --threads 2 --warmup 100 --macmimic-100k
(cd "$work" && echo '8e72574bc19499930e7f502188b96f9e2592cf8441bf3487b8d4f661b3476627  replay.bin' | sha256sum -c -)
(cd "$work" && sha256sum build/sprite_can_shadow build/sprite_supermimic_protected_ipc \
  icm/sprite_icm45686 "$runtime/tools/run_supermimic_commissioning.py" "$runtime/tools/action_tx_audit.py" \
  replay.bin > trial.sha256.tmp && mv trial.sha256.tmp trial.sha256)
