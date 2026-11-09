#!/usr/bin/env bash
# Edit constants here. No external parameters.
set -Ee -o pipefail
TEST_APPROVED=1
ROOT='/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_udma_weighted_multinode'
BACKEND='mte'
NODE_IP='141.61.45.118'
NODE_RANK=0
LOCAL_RANKS=1
CARD_ID=1  # User explicitly selected physical NPU 1 on each node.
RUN_ID='cross2_card1_20261105_112412_mte'
MASTER_PORT=29930
SHMEM_PORT=29931
LIB_MODE='staged'
[[ $# == 0 && "$TEST_APPROVED" == 1 ]] || { echo 'Test gate closed: review constants first.' >&2; exit 2; }
while IFS= read -r name; do
    case "$name" in MOONEP*|ASCEND_MOONEP*|VLLM*) unset "$name";; esac
done < <(compgen -e)
unset PYTHONPATH LD_PRELOAD ASCEND_VISIBLE_DEVICES
source /usr/local/Ascend/cann/set_env.sh
export PATH="/usr/local/python3.11.10/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/python3.11.10/lib:$LD_LIBRARY_PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export GLOO_SOCKET_IFNAME=enp34s0f1 ASCEND_MOONEP_SKIP_DUP=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY=127.0.0.1,localhost,141.61.45.118,141.61.45.124
export no_proxy="$NO_PROXY"
CARDS=$(python - "$LOCAL_RANKS" "$NODE_IP" "$CARD_ID" <<'CHECK'
import datetime,fcntl,re,socket,struct,subprocess,sys
from zoneinfo import ZoneInfo
assert datetime.datetime.now(ZoneInfo("Asia/Shanghai"))>=datetime.datetime(2026,11,5,4,tzinfo=ZoneInfo("Asia/Shanghai")),"Not before 2026-11-05 04:00"
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
    packed=fcntl.ioctl(sock.fileno(),0x8915,struct.pack('256s',b'enp34s0f1'))
assert socket.inet_ntoa(packed[20:24])==sys.argv[2],"Wrong physical node"
needed=int(sys.argv[1]);card=int(sys.argv[3]);assert needed==1 and card==1
text=subprocess.check_output(["npu-smi","info"],text=True)
free=sorted(set(map(int,re.findall(r"No running processes found in NPU (\d+)",text))))
assert card in free,("Requested physical card is not idle",card,free)
print(card)
CHECK
)
export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
OUT="$ROOT/results/$RUN_ID"
mkdir -p "$OUT"
exec 9>"$ROOT/.small_test_node$NODE_RANK.lock"
flock -n 9 || { echo 'An owned test is active.' >&2; exit 2; }
[[ ! -e "$OUT/started_node$NODE_RANK.json" ]] || { echo 'Use a fresh RUN_ID.' >&2; exit 2; }
python - "$OUT" "$NODE_RANK" "$NODE_IP" "$CARDS" "$$" "$MASTER_PORT" "$SHMEM_PORT" <<'RECORD'
import datetime,json,socket,sys
from pathlib import Path
out,node,ip,cards,pid,port,shmem=sys.argv[1:]
if node=="0":
    for number in (port,shmem):
        with socket.socket() as s:s.bind((ip,int(number)))
(Path(out)/f"started_node{node}.json").write_text(json.dumps(dict(node_ip=ip,cards=cards,owned_shell_pid=int(pid),time=datetime.datetime.now().astimezone().isoformat()),indent=2))
RECORD
set --
[[ "$LIB_MODE" == staged || "$LIB_MODE" == installed ]]
[[ "$LIB_MODE" != installed ]] || set -- --installed
cd "$ROOT/tests"
# Timeout controls only this newly created child process group.
timeout --signal=TERM --kill-after=15 240 python -m torch.distributed.run \
    --nnodes=2 --nproc-per-node="$LOCAL_RANKS" --node-rank="$NODE_RANK" \
    --master-addr=141.61.45.118 --master-port="$MASTER_PORT" \
    test_cross_node_weighted.py --backend "$BACKEND" --node-ip "$NODE_IP" \
    --master-ip 141.61.45.118 --shmem-port "$SHMEM_PORT" --output "$OUT" "$@" \
    >"$OUT/node$NODE_RANK.log" 2>&1 &
owned_pid=$!
printf '%s\n' "$owned_pid" >"$OUT/owned_launcher_node$NODE_RANK.pid"
ps -o pgid= -p "$owned_pid" >"$OUT/owned_launcher_node$NODE_RANK.pgid"
wait "$owned_pid"
