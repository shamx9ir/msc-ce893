"""Download and unpack the BIRD dev set (about 350 MB zipped, 1.4 GB unpacked) into data/bird.

Usage: python scripts/download_bird.py
"""
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
DATA = Path("data")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    zip_path = DATA / "dev.zip"
    if not zip_path.exists():
        print(f"Downloading {URL} ...")
        with urllib.request.urlopen(URL) as r, open(zip_path, "wb") as f:
            shutil.copyfileobj(r, f)
    print("Unpacking ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA)
    inner = next(DATA.glob("dev_*"))  # e.g. data/dev_20240627
    with zipfile.ZipFile(inner / "dev_databases.zip") as z:
        z.extractall(inner)
    target = DATA / "bird"
    if target.exists() or target.is_symlink():
        print(f"{target} already exists; leaving it as is")
    else:
        inner.rename(target)
    print("Done:", target.resolve())


if __name__ == "__main__":
    main()
