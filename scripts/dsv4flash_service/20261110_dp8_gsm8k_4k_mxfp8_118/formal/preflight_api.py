"""Check dataset identity, API tokenization and one real one-token response."""
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


def main():
    host, port, model, manifest_path, output_dir = sys.argv[1:]
    manifest = json.loads(Path(manifest_path).read_text())
    dataset = Path(manifest["dataset"])
    tokenizer = Path(manifest["tokenizer"])
    for path, expected in (
        (dataset, manifest["dataset_sha256"]),
        (dataset.with_name("train.jsonl"), manifest["train_sha256"]),
        (tokenizer, manifest["tokenizer_sha256"]),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Input changed after review: {path}")

    opener = build_opener(ProxyHandler({}))

    def call(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(
            f"http://{host}:{port}{path}", data=body,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=120) as response:
            return json.load(response)

    actual_models = [row["id"] for row in call("/v1/models")["data"]]
    if model not in actual_models:
        raise ValueError(f"Expected service {model}, found {actual_models}")
    questions = [json.loads(line)["question"] for line in dataset.read_text().splitlines()]
    samples = manifest["samples"]
    shortest = min(samples, key=lambda row: row["prompt_tokens"])
    longest = max(samples, key=lambda row: row["prompt_tokens"])
    for sample in (shortest, longest):
        count = call("/tokenize", dict(model=model, prompt=questions[sample["data_id"]],
                                       add_special_tokens=True))["count"]
        if count != sample["prompt_tokens"]:
            raise ValueError(f"Tokenization mismatch for question {sample['data_id']}: {count}")
    result = call("/v1/completions", dict(
        model=model, prompt=questions[longest["data_id"]], max_tokens=1, stream=False,
        temperature=0, ignore_eos=True, seed=1024,
    ))
    usage = result["usage"]
    if usage["prompt_tokens"] != longest["prompt_tokens"] or usage["completion_tokens"] != 1:
        raise ValueError(f"Unexpected preflight token counts: {usage}")
    report = dict(model=model, data_id=longest["data_id"], usage=usage, formal_measurement=False)
    (Path(output_dir) / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PREFLIGHT_OK", report)


if __name__ == "__main__":
    main()
