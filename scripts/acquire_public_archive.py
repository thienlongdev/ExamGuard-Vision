"""Download a public ZIP once and safely extract without overwriting originals."""
import argparse
from pathlib import Path
import zipfile
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "original_download.zip"
    if not archive.exists():
        partial = root / "original_download.zip.partial"
        with requests.get(args.url, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if not zipfile.is_zipfile(partial):
            raise ValueError("Response is not a ZIP archive; retained partial for inspection")
        partial.rename(archive)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if not target.exists():
                source.extract(member, root)
        print(f"Archive: {archive}; members: {len(source.infolist())}")


if __name__ == "__main__":
    main()
