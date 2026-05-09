
"""
Corpus loader: load and cache test images from the manifest.

Usage
-----
    loader = CorpusLoader(manifest_path="corpus/manifest.json",
                          corpus_root="corpus/")
    for entry in loader.iter_images():
        img = entry.image          # PIL.Image.Image
        meta = entry.metadata      # dict from manifest
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


@dataclass
class CorpusEntry:
    """A single image entry from the corpus."""

    image_id: str
    subset: str          # e.g. "photographs" (value from manifest "subset" field)
    path: Path
    metadata: dict       # raw manifest dict for this image
    image: Image.Image   # loaded PIL image


class CorpusLoader:
    """Load images listed in a corpus manifest JSON file.

    Parameters
    ----------
    manifest_path:
        Path to ``corpus/manifest.json``.
    corpus_root:
        Root directory that contains the per-subset subdirectories.
    subset:
        If given, only load images from this subset.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        corpus_root: str | Path,
        subset: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.corpus_root = Path(corpus_root)
        self.subset_filter = subset

        with open(self.manifest_path) as f:
            self._manifest: list[dict] = json.load(f)

    def iter_images(self) -> Iterator[CorpusEntry]:
        """Yield CorpusEntry objects for each loadable image in the manifest."""
        for entry in self._manifest:
            subset = entry.get("subset", "unknown")
            if self.subset_filter and subset != self.subset_filter:
                continue

            image_id = entry["id"]
            rel_path = entry["filename"]
            full_path = self.corpus_root / subset / rel_path

            if not full_path.exists():
                logger.warning("Image not found, skipping: %s", full_path)
                continue

            try:
                with Image.open(full_path) as raw:
                    img = raw.convert("RGB").copy()
            except UnidentifiedImageError:
                logger.warning("Cannot open image, skipping: %s", full_path)
                continue

            yield CorpusEntry(
                image_id=image_id,
                subset=subset,
                path=full_path,
                metadata=entry,
                image=img,
            )

    def __len__(self) -> int:
        entries = self._manifest
        if self.subset_filter:
            entries = [e for e in entries if e.get("subset") == self.subset_filter]
        return len(entries)
