#!/usr/bin/env bash
set -Eeuo pipefail
EXPERIMENT_DIR='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_dp16_optimized_retest'
export PATH='/usr/local/python3.11.10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
export HF_HOME="$EXPERIMENT_DIR/data/client_cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR='/dev/shm/optimized_dp16_20261106'
mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR"
exec python - "$EXPERIMENT_DIR" <<'PY'
import json,hashlib,sys
from pathlib import Path
from datasets import load_dataset
root=Path(sys.argv[1]);p=root/"data/longbenchv2_all503_midcut128k.json"
data=load_dataset("json",data_files=str(p.resolve()))
assert len(data["train"])==503
manifest=json.loads((root/"data/full503_manifest.json").read_text())
record=dict(samples=503,data_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((root/"data/full503_manifest.json").read_bytes()).hexdigest(),no_inference_requests=True,cache_prepared=True)
(root/"DATA_CACHE_READY.json").write_text(json.dumps(record,indent=2))
print("DATA_CACHE_READY",record,flush=True)
PY
