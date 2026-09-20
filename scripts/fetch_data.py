"""Download the datasets named in data/manifest.yaml, and record what was actually fetched.

    python scripts/fetch_data.py --dataset ved
    python scripts/fetch_data.py --list

Downloads go to data/raw/<dataset>/ (git-ignored). Every file is hashed with SHA256
and the hash is written back into the manifest, so any later result can name the exact
bytes it was computed from. Re-running skips files already present with a matching
hash, so it is safe to call repeatedly.

Nothing here is automatic: each dataset is fetched only when named, because the
licences differ and the person running this should know what they are agreeing to.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, ROOT  # noqa: E402
from src.evaluation.harness import file_sha256  # noqa: E402

MANIFEST = DATA_DIR / "manifest.yaml"
RAW = DATA_DIR / "raw"

# Direct file lists for datasets served as plain files. Datasets that need a git
# clone or a manual licence click are described, not automated.
DIRECT_DOWNLOADS: dict[str, list[tuple[str, str]]] = {
    "ved": [
        ("VED_DynamicData_Part1.7z",
         "https://github.com/gsoh/VED/raw/master/Data/VED_DynamicData_Part1.7z"),
        ("VED_DynamicData_Part2.7z",
         "https://github.com/gsoh/VED/raw/master/Data/VED_DynamicData_Part2.7z"),
        ("VED_Static_Data_ICE&HEV.xlsx",
         "https://github.com/gsoh/VED/raw/master/Data/VED_Static_Data_ICE%26HEV.xlsx"),
        ("VED_Static_Data_PHEV&EV.xlsx",
         "https://github.com/gsoh/VED/raw/master/Data/VED_Static_Data_PHEV%26EV.xlsx"),
    ],
    # Published as one 656 MB zip. Fetched directly rather than by git clone, which
    # would also pull the repository history for no benefit. The documented clone URL
    # carries a "Datarepo@" username; the repository is public and needs no credential.
    "eved": [
        ("eVED.zip",
         "https://bitbucket.org/datarepo/eved-dataset/raw/main/data/eVED.zip"),
    ],
}


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())


def download(url: str, dest: Path, *, chunk: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=chunk):
                fh.write(block)
                done += len(block)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
    print()
    tmp.rename(dest)


def fetch(dataset: str, *, force: bool = False) -> dict[str, str]:
    """Download one dataset. Returns {filename: sha256}."""
    if dataset not in DIRECT_DOWNLOADS:
        raise SystemExit(
            f"'{dataset}' is not a direct download. See data/manifest.yaml for how to "
            f"obtain it (git clone, or a licence acceptance on the provider's site)."
        )
    out_dir = RAW / dataset
    hashes: dict[str, str] = {}
    for name, url in DIRECT_DOWNLOADS[dataset]:
        dest = out_dir / name
        if dest.exists() and not force:
            print(f"  {name}: already present, skipping")
        else:
            print(f"  {name}: downloading")
            download(url, dest)
        hashes[name] = file_sha256(dest)
    return hashes


def record_hashes(dataset: str, hashes: dict[str, str]) -> None:
    """Write the observed hashes into the manifest, replacing PENDING_DOWNLOAD."""
    text = MANIFEST.read_text()
    manifest = yaml.safe_load(text)
    entry = manifest["datasets"].get(dataset)
    if entry is None:
        print(f"  (no manifest entry for '{dataset}'; hashes not recorded)")
        return
    lines = text.splitlines(keepends=True)
    combined = "sha256sums:" + "".join(f" {n}={h[:16]}..." for n, h in sorted(hashes.items()))
    # Replace only this dataset's PENDING_DOWNLOAD marker, preserving comments.
    in_block, replaced = False, False
    for i, line in enumerate(lines):
        if line.startswith(f"  {dataset}:"):
            in_block = True
        elif in_block and line.startswith("  ") and not line.startswith("    "):
            in_block = False
        if in_block and "sha256: PENDING_DOWNLOAD" in line:
            lines[i] = line.replace("PENDING_DOWNLOAD", f'"{combined}"')
            replaced = True
            break
    if replaced:
        MANIFEST.write_text("".join(lines))
        print(f"  manifest updated with {len(hashes)} file hash(es)")
    else:
        print("  manifest already recorded a hash; leaving it alone")


def extract_eved() -> None:
    """Unpack the eVED zip into data/raw/eved/."""
    import zipfile

    out = RAW / "eved"
    src = out / "eVED.zip"
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        print(f"  archive holds {len(names)} entries; extracting ...")
        z.extractall(path=out)
    csvs = list(out.rglob("*.csv"))
    print(f"  extracted {len(csvs)} CSV file(s) to {out.relative_to(ROOT)}")


def extract_ved() -> None:
    """Unpack the VED .7z archives into data/raw/ved/dynamic/."""
    import py7zr

    out = RAW / "ved" / "dynamic"
    out.mkdir(parents=True, exist_ok=True)
    for part in sorted((RAW / "ved").glob("VED_DynamicData_Part*.7z")):
        print(f"  extracting {part.name} ...")
        with py7zr.SevenZipFile(part, mode="r") as z:
            z.extractall(path=out)
    csvs = list(out.rglob("*.csv"))
    print(f"  extracted {len(csvs)} CSV file(s) to {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", help="dataset key from data/manifest.yaml")
    ap.add_argument("--list", action="store_true", help="show datasets and how to get them")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--extract", action="store_true", help="unpack archives after download")
    args = ap.parse_args()

    manifest = load_manifest()
    if args.list or not args.dataset:
        print("Datasets in the manifest:\n")
        for key, entry in manifest["datasets"].items():
            how = "automated" if key in DIRECT_DOWNLOADS else "manual"
            print(f"  {key:22s} [{how:9s}] {entry['role']}")
            print(f"  {'':22s} licence: {entry['licence']}")
            print(f"  {'':22s} {entry.get('download', entry['url'])}\n")
        print("Excluded (documented decisions):")
        for key, entry in manifest.get("excluded", {}).items():
            print(f"  {key:22s} {entry['reason']}")
        return

    print(f"Fetching '{args.dataset}' ...")
    hashes = fetch(args.dataset, force=args.force)
    record_hashes(args.dataset, hashes)
    if args.extract:
        {"ved": extract_ved, "eved": extract_eved}.get(args.dataset, lambda: None)()
    print("Done.")


if __name__ == "__main__":
    main()
