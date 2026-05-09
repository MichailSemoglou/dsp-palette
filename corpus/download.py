
"""
Reproducible corpus downloader.

Usage
-----
    python download.py              # full corpus (90 COCO photographs)
    python download.py --dev        # dev subset only (manifest entries with "dev": true)
    python download.py --verify     # verify checksums only, no downloads
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).parent / "manifest.json"
CORPUS_ROOT = Path(__file__).parent

HEADERS = {"User-Agent": "color-analysis-tool-research/1.0 (academic use)"}
TIMEOUT_SECONDS = 30
RETRY_COUNT = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_entry(entry: dict, dry_run: bool = False) -> bool:
    """Download a single manifest entry.  Returns True on success."""
    subset = entry.get("subset", "unknown")
    filename = entry["filename"]
    url = entry.get("original_url")
    expected_sha = entry.get("sha256")

    dest = CORPUS_ROOT / subset / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_sha:
            actual = sha256_file(dest)
            if actual != expected_sha:
                logger.warning(
                    "Checksum mismatch for %s (expected %s, got %s); re-downloading",
                    filename, expected_sha[:8], actual[:8],
                )
                if not url:
                    logger.warning("No URL for %s; removing corrupted file", filename)
                    dest.unlink()
                    return False
            else:
                logger.info("OK (cached): %s", filename)
                return True
        else:
            logger.info("OK (cached, no checksum): %s", filename)
            return True

    if not url:
        logger.warning("No URL for %s; skipping", filename)
        return False

    if dry_run:
        logger.info("DRY RUN: would download %s → %s", url, dest)
        return True

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            break
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, RETRY_COUNT, filename, exc)
            if attempt < RETRY_COUNT:
                time.sleep(2**attempt)
            else:
                logger.error("Failed to download %s after %d attempts", filename, RETRY_COUNT)
                return False

    if expected_sha:
        actual = sha256_file(dest)
        if actual != expected_sha:
            logger.error(
                "Checksum mismatch after download for %s (expected %s, got %s)",
                filename, expected_sha, actual,
            )
            dest.unlink(missing_ok=True)
            return False

    logger.info("Downloaded: %s", filename)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download DSP evaluation corpus")
    parser.add_argument("--dev", action="store_true", help="Only download dev subset")
    parser.add_argument("--verify", action="store_true", help="Verify checksums only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    entries = manifest
    if args.dev:
        entries = [e for e in manifest if e.get("dev", False)]
        logger.info("Dev mode: %d images", len(entries))
    else:
        logger.info("Full corpus: %d images", len(entries))

    ok = 0
    fail = 0
    for entry in entries:
        if args.verify:
            dest = CORPUS_ROOT / entry.get("subset", "unknown") / entry["filename"]
            if dest.exists() and entry.get("sha256"):
                actual = sha256_file(dest)
                if actual == entry["sha256"]:
                    logger.info("PASS: %s", entry["filename"])
                    ok += 1
                else:
                    logger.error("FAIL (checksum): %s", entry["filename"])
                    fail += 1
            elif not dest.exists():
                logger.warning("MISSING: %s", entry["filename"])
                fail += 1
            else:
                logger.info("SKIP (no checksum): %s", entry["filename"])
                ok += 1
        else:
            if download_entry(entry, dry_run=args.dry_run):
                ok += 1
            else:
                fail += 1

    logger.info("Done: %d OK, %d failed", ok, fail)
    if fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
