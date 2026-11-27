#!/usr/bin/env bash
# Native Bianbu build and file-only regression tests.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(uname -m) == riscv64 ]]
w=sim2real/build/k3
inc=${SPRITE_ORT_INCLUDE:-/usr/include/onnxruntime}
lib=${SPRITE_ORT_LIBRARY:-/usr/lib/riscv64-linux-gnu/libonnxruntime.so}
export PYTHONPATH="$PWD/runtime/src" OPENBLAS_NUM_THREADS=1
python3 -m open_sprite_runtime.supermimic_run002 verify --bundle policies/opensprite-velocity-100k
cmake -S runtime/native -B "$w/build" -DCMAKE_BUILD_TYPE=Release -DSPRITE_ORT_INCLUDE="$inc" -DSPRITE_ORT_LIBRARY="$lib"
cmake --build "$w/build" -j4
ctest --test-dir "$w/build" --output-on-failure
cmake -S runtime/native/icm45686 -B "$w/icm" -DCMAKE_BUILD_TYPE=Release
cmake --build "$w/icm" -j4
ctest --test-dir "$w/icm" --output-on-failure
"$w/build/sprite_supermimic_replay" policies/opensprite-velocity-100k/policy.onnx sim2real/evidence/xbox803-input.bin "$w/walk.bin" --threads 2 --warmup 100 --macmimic-100k
python3 runtime/tools/verify_k3_walk.py "$w/walk.bin" sim2real/evidence/xbox803-jetson-reference.npz
for motion in standing_groove disco_point; do
  python3 runtime/tools/verify_standing_groove.py "policies/opensprite-${motion//_/-}" "$w/build/sprite_supermimic_protected_ipc" --motion "$motion" --report "$w/$motion.json"
done
