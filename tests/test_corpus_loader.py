
"""Unit tests for evaluation/corpus_loader.py."""

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from evaluation.corpus_loader import CorpusEntry, CorpusLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(directory: Path, entries: list[dict]) -> Path:
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(entries))
    return manifest_path


def _make_rgb_image(path: Path, size: tuple[int, int] = (10, 10)) -> None:
    """Save a small solid-colour JPEG to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(100, 150, 200)).save(path)


# ---------------------------------------------------------------------------
# iter_images — happy path
# ---------------------------------------------------------------------------


def test_iter_images_yields_correct_entries():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subset_dir = root / "photographs"
        subset_dir.mkdir()
        _make_rgb_image(subset_dir / "img001.jpg")

        manifest = _write_manifest(
            root,
            [{"id": "coco_001", "subset": "photographs", "filename": "img001.jpg"}],
        )

        loader = CorpusLoader(manifest_path=manifest, corpus_root=root)
        entries = list(loader.iter_images())

    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, CorpusEntry)
    assert entry.image_id == "coco_001"
    assert entry.subset == "photographs"
    assert isinstance(entry.image, Image.Image)
    assert entry.image.mode == "RGB"


def test_iter_images_skips_missing_files(caplog):
    import logging

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = _write_manifest(
            root,
            [
                {
                    "id": "missing_001",
                    "subset": "photographs",
                    "filename": "does_not_exist.jpg",
                }
            ],
        )

        loader = CorpusLoader(manifest_path=manifest, corpus_root=root)
        with caplog.at_level(logging.WARNING, logger="evaluation.corpus_loader"):
            entries = list(loader.iter_images())

    assert entries == []
    assert any("not found" in msg.lower() or "skipping" in msg.lower() for msg in caplog.messages)


def test_iter_images_filters_by_subset():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "photos").mkdir()
        (root / "paintings").mkdir()
        _make_rgb_image(root / "photos" / "a.jpg")
        _make_rgb_image(root / "paintings" / "b.jpg")

        manifest = _write_manifest(
            root,
            [
                {"id": "p1", "subset": "photos", "filename": "a.jpg"},
                {"id": "p2", "subset": "paintings", "filename": "b.jpg"},
            ],
        )

        loader = CorpusLoader(manifest_path=manifest, corpus_root=root, subset="photos")
        entries = list(loader.iter_images())

    assert len(entries) == 1
    assert entries[0].image_id == "p1"


def test_iter_images_multiple_entries():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subset_dir = root / "test"
        subset_dir.mkdir()
        for i in range(3):
            _make_rgb_image(subset_dir / f"img{i:03d}.jpg")

        entries_manifest = [
            {"id": f"img_{i}", "subset": "test", "filename": f"img{i:03d}.jpg"}
            for i in range(3)
        ]
        manifest = _write_manifest(root, entries_manifest)

        loader = CorpusLoader(manifest_path=manifest, corpus_root=root)
        entries = list(loader.iter_images())

    assert len(entries) == 3
    ids = [e.image_id for e in entries]
    assert set(ids) == {"img_0", "img_1", "img_2"}


def test_len_reflects_manifest_size():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = _write_manifest(
            root,
            [
                {"id": "a", "subset": "s1", "filename": "a.jpg"},
                {"id": "b", "subset": "s1", "filename": "b.jpg"},
                {"id": "c", "subset": "s2", "filename": "c.jpg"},
            ],
        )
        loader_all = CorpusLoader(manifest_path=manifest, corpus_root=root)
        loader_s1 = CorpusLoader(manifest_path=manifest, corpus_root=root, subset="s1")

    assert len(loader_all) == 3
    assert len(loader_s1) == 2
