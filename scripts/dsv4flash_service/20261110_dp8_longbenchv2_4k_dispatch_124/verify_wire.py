"""Host-only checks for the staged or installed MXFP8 wire library."""
import ctypes
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGED = ROOT / "native/libascend_moonep_mxfp8_wire.so"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path):
    expected = (STAGED.parent / "SHA256SUMS").read_text().split()[0]
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"Wire library {path} does not match the staged build")
    library = ctypes.CDLL(str(path))
    library.moonep_mxfp8_wire_abi.restype = ctypes.c_int
    if library.moonep_mxfp8_wire_abi() != 1 or not library.moonep_mxfp8_wire_bounded:
        raise RuntimeError("Expected wire ABI 1 and the bounded unpack symbol")
    print(f"Verified wire ABI 1 and bounded unpack, sha256={actual}")


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve(strict=True))
