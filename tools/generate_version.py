import re
import subprocess
from pathlib import Path

from with_argparse import with_argparse

UNKNOWN = "Unknown"
RELEASE_PATTERN = re.compile(r"/v[0-9]+(\.[0-9]+)*(-rc[0-9]+)?/")


# amended from
# https://github.com/pytorch/pytorch/blob/87d46d70d7754e32eb0e6689688f4336e4e7c955/tools/generate_torch_version.py#L99
def get_library_version() -> str:
    library_root = Path(__file__).parent.parent
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--exact"],
            cwd=library_root,
            encoding="ascii",
            capture_output=True,
        ).stdout.strip()
        if RELEASE_PATTERN.match(tag):
            return tag
        else:
            return UNKNOWN
    except Exception:
        return UNKNOWN


def generate_version():
    library_root = Path(__file__).parent.parent
    version_file = library_root / "safetensors_dataset/version.py"
    version = get_library_version()
    if version == UNKNOWN:
        print("Cannot create version file, no tag set for build ...")
        exit(1)
    with open(version_file, "w") as f:
        f.write(f"__version__ = {version}\n")


if __name__ == "__main__":
    generate_version()
