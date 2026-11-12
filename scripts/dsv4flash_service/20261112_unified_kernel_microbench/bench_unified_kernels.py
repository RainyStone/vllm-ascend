#!/usr/bin/env python3
"""One-card split vs unified device-library benchmark; no model or SHMEM init."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu


ROOT = Path(__file__).resolve().parent
SDK = ROOT.parents[3] / "ascend-moonep"


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def measured(function, iterations, batches, warmup):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    device, wall = [], []
    for _ in range(batches):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - start) * 1e6 / iterations)
        device.append(begin.elapsed_time(end) * 1000 / iterations)
    return {
        "device_us": device, "wall_us": wall,
        "median_device_us": statistics.median(device),
        "median_wall_us": statistics.median(wall),
        "iterations_per_batch": iterations, "batches": batches, "warmup": warmup,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("split", "unified"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--only-case", default=None)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    dev = torch.device("npu", args.device)
    torch.manual_seed(20261112)
    torch.npu.manual_seed_all(20261112)
    generator = torch.Generator().manual_seed(20261112)
    result = {"variant": args.variant, "host": os.uname().nodename,
              "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "logical_device": args.device, "cases": {}, "correctness": {},
              "scope": "local wire/publish/Home GEMM1 only; no communication or model"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    wire_module = module("unified_bench_wire", SDK / "ascend_moonep/mxfp8_transport.py")
    publish_module = module("unified_bench_publish", SDK / "ascend_moonep/combine_publish.py")
    home_module = module("unified_bench_home", SDK / "ascend_moonep/home_gmm.py")
    artifacts = ROOT / "artifacts"
    if args.variant == "split":
        paths = {key: artifacts / "split" / name for key, name in {
            "wire": "libascend_moonep_mxfp8_wire.so",
            "publish": "libascend_moonep_publish.so",
            "home": "libascend_moonep_home_gmm.so"}.items()}
    else:
        paths = {key: artifacts / "unified/libascend_moonep_kernels.so"
                 for key in ("wire", "publish", "home")}
    result["libraries"] = {key: {"path": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for key, path in paths.items()}
    start = time.perf_counter()
    wire = wire_module.Mxfp8Wire(paths["wire"])
    publisher = publish_module.CombinePublisher(paths["publish"])
    home = home_module.HomeGmmRange(paths["home"])
    result["load_helpers_ms"] = (time.perf_counter() - start) * 1000
    result["home_gmm_cores"] = home.cores
    result["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    maps = Path("/proc/self/maps").read_text()
    mapped = sorted({line.split()[-1] for line in maps.splitlines()
                     if "libascend_moonep_" in line})
    assert set(mapped) == set(map(str, paths.values())), mapped
    result["mapped_moonep_libraries"] = mapped

    def persist():
        output.write_text(json.dumps(result, indent=2))

    def check(name, condition, **metadata):
        assert condition, (name, metadata)
        result["correctness"][name] = dict(passed=True, **metadata)

    def time_case(name, function, **metadata):
        values = measured(function, args.iterations, args.batches, args.warmup)
        result["cases"][name] = dict(values, **metadata)
        print(args.variant, name, round(values["median_device_us"], 3), "us", flush=True)
        persist()

    if args.only_case is None:
        hidden, rows = 4096, 4096 * 6
        for count in (16, 512, 4096):
            q = torch.randint(0, 256, (count, hidden), dtype=torch.uint8,
                              generator=generator).to(dev).view(torch.float8_e4m3fn)
            scales = torch.randint(0, 256, (count, hidden // 64, 2), dtype=torch.uint8,
                                   generator=generator).to(dev)
            packet = torch.empty(count, hidden // 2 + hidden // 64,
                                 dtype=torch.bfloat16, device=dev)
            wire.pack(q, scales, packet)
            expected = torch.cat((q.view(torch.uint8), scales.reshape(count, -1)), dim=1)
            check("wire_pack_" + str(count), torch.equal(packet.view(torch.uint8), expected),
                  digest=digest(packet))
            time_case("wire_pack_" + str(count), lambda: wire.pack(q, scales, packet),
                      rows=count, hidden=hidden)
        del q, scales, packet, expected
        packet = torch.randint(0, 256, (rows, hidden + hidden // 32), dtype=torch.uint8,
                               generator=generator).to(dev).view(torch.bfloat16)
        activation = torch.full((rows, hidden), 0xAA, dtype=torch.uint8, device=dev)
        scales = torch.full((rows, hidden // 64, 2), 0xAA, dtype=torch.uint8, device=dev)
        q = activation.view(torch.float8_e4m3fn)
        wire.unpack(packet, q, scales)
        check("wire_full", torch.equal(activation, packet.view(torch.uint8)[:, :hidden])
              and torch.equal(scales.reshape(rows, -1), packet.view(torch.uint8)[:, hidden:]),
              digest=digest(activation))
        time_case("wire_unpack_full", lambda: wire.unpack(packet, q, scales), rows=rows)
        for bound in (0, 128, 512, 3072, rows, rows + 1000):
            activation.fill_(0xAA)
            scales.fill_(0xAA)
            row_bound = torch.tensor([bound], dtype=torch.int32, device=dev)
            wire.unpack_bounded(packet, q, scales, row_bound)
            limit = min(rows, bound + 128)
            check("wire_bounded_" + str(bound),
                  torch.equal(activation[:limit], packet.view(torch.uint8)[:limit, :hidden])
                  and torch.equal(scales[:limit].reshape(limit, -1),
                                  packet.view(torch.uint8)[:limit, hidden:])
                  and bool((activation[limit:] == 0xAA).all().item())
                  and bool((scales[limit:] == 0xAA).all().item()),
                  rows_written=limit)
            if bound in (128, 512, 3072):
                time_case("wire_unpack_bounded_" + str(bound),
                          lambda: wire.unpack_bounded(packet, q, scales, row_bound),
                          capacity=rows, bound=bound, rows_written=limit)
        del packet, activation, scales, q, row_bound
        torch.npu.empty_cache()

        for capacity, valid in ((3072, 128), (rows, 3072), (rows, rows)):
            source = torch.full((capacity, hidden), float("nan"),
                                dtype=torch.bfloat16, device=dev)
            source[:valid] = 0.5
            destination = torch.full_like(source, -123)
            counts = torch.tensor([valid // 64] * 64, dtype=torch.int64, device=dev)
            counts[-1] += valid % 64
            publisher.publish(source, destination, counts)
            check("publish_" + str(capacity) + "_" + str(valid),
                  bool((destination[:valid] == 0.5).all().item())
                  and bool((destination[valid:] == 0).all().item())
                  and bool(destination.isfinite().all().item()),
                  digest=digest(destination))
            time_case("publish_" + str(capacity) + "_" + str(valid),
                      lambda: publisher.publish(source, destination, counts),
                      capacity=capacity, valid=valid, hidden=hidden)
            del source, destination, counts
        torch.npu.empty_cache()

    def native(x, weight, counts, weight_scale=None, x_scale=None):
        kwargs = dict(x=[x], weight=[weight], group_list=counts, group_list_type=1,
                      split_item=2, group_type=0, output_dtype=torch.bfloat16)
        if weight_scale is not None:
            kwargs.update(antiquant_scale=[weight_scale], per_token_scale=[x_scale],
                          per_token_scale_dtype=torch.float8_e8m0fnu,
                          x_dtype=torch.float8_e4m3fn,
                          weight_dtype=torch_npu.float4_e2m1fn_x2)
        return torch_npu.npu_grouped_matmul(**kwargs)[0]

    for kind in ("bf16", "mxfp4"):
        for label, counts_host in (
            ("small", [17, 0, 31, 9] * 8),
            ("prefill", [129, 0, 257, 126] * 8),
        ):
            if args.only_case is not None and not args.only_case.startswith("gmm_" + kind + "_" + label + "_"):
                continue
            groups, k, n, split = 32, 4096, 4096, 16
            valid = sum(counts_host)
            capacity = valid + 13
            x = (torch.randn(capacity, k, dtype=torch.bfloat16,
                             generator=generator) * 0.25).to(dev)
            counts = torch.tensor(counts_host, dtype=torch.int64, device=dev)
            if kind == "bf16":
                weight = (torch.randn(groups, k, n, dtype=torch.bfloat16,
                                      generator=generator) * 0.125).to(dev)
                invoke = lambda lo, hi: home.bf16(x, weight, counts, out, lo, hi)
                reference = native(x, weight, counts)
            else:
                raw = torch.randint(0, 256, (groups, n, k // 2), dtype=torch.uint8,
                                    generator=generator).to(dev)
                weight = torch_npu.npu_format_cast(
                    raw, 29, customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2).transpose(1, 2)
                del raw
                weight_scale = torch.full((groups, n, k // 64, 2), 119,
                                          dtype=torch.uint8, device=dev).transpose(-3, -2)
                x, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
                invoke = lambda lo, hi: home.mxfp4(
                    x, weight, weight_scale, x_scale, counts, out, lo, hi)
                reference = native(x, weight, counts, weight_scale, x_scale)
            out = torch.full((capacity, n), -123, dtype=torch.bfloat16, device=dev)
            boundary = sum(counts_host[:split])
            invoke(0, split)
            torch.npu.synchronize()
            check(kind + "_" + label + "_home_range",
                  bool((out[boundary:] == -123).all().item()))
            snapshot = out.clone()
            invoke(split, split)
            torch.npu.synchronize()
            check(kind + "_" + label + "_empty_range", torch.equal(out, snapshot))
            del snapshot
            invoke(split, groups)
            torch.npu.synchronize()
            a, b = out[:valid].float(), reference[:valid].float()
            relative_l2 = float((a - b).norm().item() / max(b.norm().item(), 1e-12))
            torch.testing.assert_close(a, b, rtol=0.015, atol=0.125)
            check(kind + "_" + label + "_native",
                  bool(a.isfinite().all().item()) and relative_l2 <= 0.004
                  and bool((out[valid:] == -123).all().item()),
                  relative_l2=relative_l2, digest=digest(out))
            for range_name, begin, end in (("home", 0, split), ("slot", split, groups),
                                           ("full", 0, groups)):
                case_name = "gmm_" + kind + "_" + label + "_" + range_name
                if args.only_case is not None and args.only_case != case_name:
                    continue
                time_case(case_name, lambda: invoke(begin, end), groups=groups, k=k, n=n,
                          valid_rows=valid, begin=begin, end=end)
            del x, weight, counts, out, reference, a, b
            if kind == "mxfp4":
                del weight_scale, x_scale
            torch.npu.empty_cache()

    assert result["cases"], "No matching benchmark case"
    result["correctness_pass"] = all(x["passed"] for x in result["correctness"].values())
    result["finished"] = True
    persist()
    print("FINISHED", args.variant, "cases", len(result["cases"]),
          "checks", len(result["correctness"]), flush=True)


if __name__ == "__main__":
    main()
