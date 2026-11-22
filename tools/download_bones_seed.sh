#!/usr/bin/env bash
# BONES_SOURCE=modelscope selects the mirror. Both routes verify the official revision.
set -euo pipefail
destination=${1:-data/bones-seed}
revision=2f59b2077b9da34dd4e43618e705c7cb962c9a66
source_name=${BONES_SOURCE:-huggingface}
files=(LICENSE.md README.md metadata/seed_metadata_v004.csv g1.tar.gz)
case "$source_name" in
  huggingface)
    HF_ENDPOINT=https://huggingface.co hf download bones-studio/seed "${files[@]}" \
      --repo-type dataset --revision "$revision" --local-dir "$destination"
    ;;
  modelscope)
    mkdir -p "$destination/metadata"
    for name in "${files[@]}"; do
      target="$destination/$name"
      if [[ ! -f "$target" ]]; then
        curl -fLsS --retry 3 -C - -o "$target.part" \
          "https://modelscope.cn/datasets/bones-studio/seed/resolve/master/$name"
        mv "$target.part" "$target"
      fi
    done
    ;;
  *)
    echo "Expected BONES_SOURCE=huggingface or modelscope" >&2
    exit 2
    ;;
esac
"${PYTHON:-python3}" "$(dirname "$0")/verify_bones_seed.py" "$destination" --source "$source_name"
