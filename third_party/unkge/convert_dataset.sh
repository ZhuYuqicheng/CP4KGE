#!/usr/bin/env bash
# Downloads, converts and verifies the authors' models of one dataset, then deletes the
# downloaded files (see README.md).
#
#   bash third_party/unkge/convert_dataset.sh <dataset> <Models.zip or its URL> <UnKGCP checkout>
set -euo pipefail
shopt -s nullglob
dataset=$1 zip=$2 unkgcp=$3
cd "$(dirname "$0")/../.."

echo "== $dataset: download"
uv run python third_party/unkge/download.py models --zip "$zip" --datasets "$dataset"
echo "== $dataset: convert"
TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 \
  uv run --with tensorflow python third_party/unkge/convert.py --datasets "$dataset"
echo "== $dataset: verify"
uv run python third_party/unkge/verify.py --datasets "$dataset" --beurre-code "$unkgcp/beurre-main" \
  | tee "models/unkgcp/verify-$dataset.md"

folders=(models/unkgcp/raw/"$dataset"-*/seed*)
for folder in ${folders[@]+"${folders[@]}"}; do
  [ -e "models/unkgcp/$(basename "$(dirname "$folder")")/$(basename "$folder").pt" ] \
    || { echo "not converted: $folder"; exit 1; }
done
echo "== $dataset: ${#folders[@]} downloaded runs converted and verified; deleting them"
rm -rf models/unkgcp/raw/"$dataset"-*
echo "== $dataset: done"
