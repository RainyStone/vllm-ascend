"""Record diagnostic collection boundaries without changing the worker state."""
import json
import re
import sys
import time
from pathlib import Path


def main():
    folder = Path(sys.argv[1])
    point = sys.argv[2]
    assert point in ("before", "after")
    audit_path, = folder.glob("*_mode_audit.json")
    audit = json.loads(audit_path.read_text())
    assert audit["phase"] == "diagnostic" and audit["valid"]
    directory = Path(audit["rank_stats_dir"])
    snapshots = {}
    for path in sorted(directory.glob("moe_par_*_rank*.log")):
        with path.open("rb") as stream:
            header = stream.readline()
            rank = re.search(rb"\bep_rank=(\d+)", header)
            if not rank:
                raise ValueError(f"Missing EP rank in {path}")
            stream.seek(0, 2)
            size = stream.tell()
            # Work from the end only. The offset remains an exact persisted-byte
            # boundary; the step field describes just the retained tail.
            stream.seek(max(0, size - 512 * 1024))
            steps = re.findall(rb"(?:^|\n)step=(\d+) ", stream.read())
        ep = int(rank[1])
        if ep in snapshots:
            raise ValueError(f"Duplicate EP rank {ep}")
        snapshots[ep] = dict(path=str(path), bytes=size, tail_max_step=max(map(int, steps)) if steps else None)
    if point == "after" and set(snapshots) != set(range(8)):
        raise ValueError(f"Expected eight rank logs, got {sorted(snapshots)}")
    result = dict(at_unix=time.time(), point=point, phase="diagnostic",
                  flush_every_steps=1, directory=str(directory), ranks=snapshots)
    (folder/f"rank_boundary_{point}.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
