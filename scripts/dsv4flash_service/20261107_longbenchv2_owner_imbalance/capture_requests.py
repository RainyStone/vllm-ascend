"""Submit checked LongBench-v2 prompts and validate per-request owner counts."""
import argparse
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parent
BASE = ROOT.with_name("20261106_dp16_optimized_retest") / "data"
WORKSPACE = ROOT.parents[3]
TEMPLATE = WORKSPACE / "benchmark/ais_bench/benchmark/configs/datasets/longbenchv2/longbenchv2_gen_75fbba.py"
MODEL = "dsv4flash_dp16_owner_capture_off"
URL = "http://141.61.45.118:8445/v1/completions"


def load_dataset():
    manifest = json.loads((BASE / "full503_manifest.json").read_text())
    prepared = (BASE / "longbenchv2_all503_midcut128k.json").read_bytes()
    assert hashlib.sha256(prepared).hexdigest() == manifest["prepared_sha256"]
    tree = ast.parse(TEMPLATE.read_text())
    templates = [node.value.value for node in ast.walk(tree)
                 if isinstance(node, ast.keyword) and node.arg == "prompt"
                 and isinstance(node.value, ast.Constant)]
    assert len(templates) == 1
    data = json.loads(prepared)
    prompts = {}
    for item in manifest["samples"]:
        prompt = templates[0].format(**data[item["source_index"]])
        assert hashlib.sha256(prompt.encode()).hexdigest() == item["prompt_sha256"]
        prompts[item["data_id"]] = (item, prompt)
    assert len(prompts) == 503
    return manifest, prompts


def submit(tag, ordinal, sample, prompt):
    name = f"{tag}-{ordinal:04d}"
    request = Request(
        URL,
        data=json.dumps(dict(model=MODEL, prompt=prompt, max_tokens=1,
                             ignore_eos=True, temperature=0, seed=1024)).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": name},
    )
    with build_opener(ProxyHandler({})).open(request, timeout=1800) as response:
        result = json.load(response)
    assert result["model"] == MODEL and result["id"] == "cmpl-" + name
    usage = result["usage"]
    assert usage["prompt_tokens"] == sample["prompt_tokens"], (sample["data_id"], usage)
    assert usage["completion_tokens"] == 1, usage
    return dict(ordinal=ordinal, data_id=sample["data_id"], prompt_tokens=usage["prompt_tokens"],
                prompt_sha256=sample["prompt_sha256"], response_id=result["id"])


def count_routes(tag, receipts):
    steps = defaultdict(list)
    for path in sorted((ROOT / "data/routes").glob("rank*.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["tag"] == tag:
                    steps[row["request"]].append(row)
    assert set(steps) == set(range(len(receipts))), (
        "missing or extra request routing", len(steps), len(receipts),
        sorted(set(range(len(receipts))) - set(steps))[:12],
    )
    report = []
    for receipt in receipts:
        rows = sorted(steps[receipt["ordinal"]], key=lambda row: row["start"])
        assert len({row["source_rank"] for row in rows}) == 1
        cursor, counts = 0, [0] * 16
        for row in rows:
            assert row["start"] == cursor, (receipt, cursor, row)
            chunk = row["owner_pairs"]
            assert len(chunk) == 16 and all(v >= 0 for v in chunk)
            assert sum(chunk) == row["tokens"] * 43 * 6
            cursor += row["tokens"]
            counts = [a+b for a,b in zip(counts, chunk, strict=True)]
        assert cursor == receipt["prompt_tokens"], (receipt, cursor)
        report.append({**receipt, "source_rank": rows[0]["source_rank"],
                       "chunks": len(rows), "owner_pairs": counts,
                       "ratio": 16 * max(counts) / sum(counts)})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["check", "smoke", "scan", "replay"], required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    manifest, prompts = load_dataset()
    if args.phase == "check":
        print("Validated prompt count", len(prompts), flush=True)
        return
    if args.phase == "smoke":
        sample = min(prompts, key=lambda i: prompts[i][0]["prompt_tokens"])
        groups = [("lbsmoke", [sample])]
    elif args.phase == "scan":
        groups = [("lbscan", list(prompts))]
    else:
        groups = [(f"lbv{round(group['target']*10)}", group["source_data_ids"])
                  for group in json.loads(args.selection.read_text())["groups"]]

    for tag, ids in groups:
        outfile = ROOT / "data" / f"{tag}.json"
        receipts = []
        with outfile.with_suffix(".receipts.jsonl").open("x", encoding="utf-8") as output:
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(submit, tag, ordinal, *prompts[sample])
                           for ordinal, sample in enumerate(ids)]
                for future in as_completed(futures):
                    receipt = future.result()
                    receipts.append(receipt)
                    output.write(json.dumps(receipt) + "\n")
                    output.flush()
                    if len(receipts) % 16 == 0 or len(receipts) == len(ids):
                        print(tag, "responses", len(receipts), "/", len(ids), flush=True)
        receipts.sort(key=lambda row: row["ordinal"])
        # Worker flush normally precedes the response. Let the shared file
        # system expose the final worker append before validating the cohort.
        time.sleep(2)
        samples = count_routes(tag, receipts)
        totals = [sum(row["owner_pairs"][rank] for row in samples) for rank in range(16)]
        with outfile.open("x", encoding="utf-8") as output:
            json.dump(dict(prepared_sha256=manifest["prepared_sha256"], samples=samples,
                           owner_pairs=totals, ratio=16*max(totals)/sum(totals)), output, indent=2)
        print(tag, "validated", len(samples), "ratio", 16*max(totals)/sum(totals), flush=True)


if __name__ == "__main__":
    main()
