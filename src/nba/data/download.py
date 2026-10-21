"""Acquire the documented Hillstrom CSV without weakening TLS checks."""
from __future__ import annotations
import argparse, hashlib, json, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = "https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
MANUAL_PAGE = "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html"
DEFAULT_PATH = Path("data/raw/hillstrom.csv")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def acquire(path: Path = DEFAULT_PATH, timeout: int = 60) -> Path:
    """Download to *path*, preserving an existing raw file unchanged."""
    if path.exists():
        print(f"Raw file already exists; preserving it: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "nba-platform-data-loader/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            if response.status != 200 or not payload:
                raise OSError(f"Unexpected HTTP response {response.status} or empty file")
        if b"," not in payload[:4096] or b"<html" in payload[:1024].lower():
            raise ValueError("Response does not appear to be the expected CSV")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not download the documented CSV ({exc}). Open {MANUAL_PAGE}, follow its dataset link, "
            f"and save the original file to {path.resolve()}. TLS verification was not disabled; "
            "no synthetic data was substituted."
        ) from exc
    path.write_bytes(payload)
    provenance = {"dataset": "Hillstrom MineThatData E-Mail Analytics Challenge", "source_url": SOURCE_URL,
        "source_page": MANUAL_PAGE, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_file(path), "bytes": path.stat().st_size}
    path.with_suffix(path.suffix + ".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Downloaded {path} ({provenance['bytes']:,} bytes; SHA-256 {provenance['sha256']})")
    return path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--check", action="store_true", help="check for a local CSV without downloading")
    args = parser.parse_args()
    if args.check:
        if not args.path.is_file():
            raise SystemExit(f"Dataset not found at {args.path.resolve()}. Download manually from {MANUAL_PAGE}.")
        print(f"Found {args.path.resolve()} ({args.path.stat().st_size:,} bytes; SHA-256 {sha256_file(args.path)})")
    else:
        try:
            acquire(args.path)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

if __name__ == "__main__":
    main()
