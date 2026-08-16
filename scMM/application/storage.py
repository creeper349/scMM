"""Safe access to server-mounted data directories."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_RAW_SUFFIXES = frozenset({".mzml", ".mzxml"})


@dataclass(frozen=True)
class StorageRoot:
    """A named directory that the UI is allowed to browse."""

    label: str
    path: Path

    def __post_init__(self) -> None:
        label = self.label.strip()
        if not label:
            raise ValueError("Storage root label cannot be empty")
        path = Path(self.path).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(path)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class StorageEntry:
    """One directory or supported raw-data file below a storage root."""

    name: str
    relative_path: Path
    is_directory: bool
    size_bytes: int | None


class StorageCatalog:
    """Resolve and enumerate paths without allowing escape from configured roots."""

    def __init__(self, roots: Iterable[StorageRoot]) -> None:
        roots_by_label: dict[str, StorageRoot] = {}
        for root in roots:
            if root.label in roots_by_label:
                raise ValueError(f"Duplicate storage root label: {root.label}")
            roots_by_label[root.label] = root
        if not roots_by_label:
            raise ValueError("At least one storage root is required")
        self._roots = roots_by_label

    @property
    def roots(self) -> tuple[StorageRoot, ...]:
        """Return configured roots in display order."""
        return tuple(self._roots.values())

    def root(self, label: str) -> StorageRoot:
        """Look up a configured root by its display label."""
        try:
            return self._roots[label]
        except KeyError as exc:
            raise KeyError(f"Unknown storage root: {label}") from exc

    def resolve(self, root_label: str, selected_path: str | Path) -> Path:
        """Resolve a selected path and reject traversal or symlink escapes."""
        root = self.root(root_label)
        candidate = Path(selected_path).expanduser()
        if not candidate.is_absolute():
            candidate = root.path / candidate
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root.path):
            raise PermissionError(f"Path is outside storage root {root.label!r}: {resolved}")
        return resolved

    def resolve_raw_file(self, root_label: str, selected_path: str | Path) -> Path:
        """Resolve a readable mzML/mzXML file below a configured root."""
        resolved = self.resolve(root_label, selected_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Not a file: {resolved}")
        if resolved.suffix.lower() not in SUPPORTED_RAW_SUFFIXES:
            raise ValueError("Only mzML and mzXML raw-data files are supported")
        return resolved

    def list_entries(
        self,
        root_label: str,
        directory: str | Path = ".",
    ) -> tuple[StorageEntry, ...]:
        """List child directories and supported raw-data files."""
        root = self.root(root_label)
        resolved = self.resolve(root_label, directory)
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        entries: list[StorageEntry] = []
        for child in resolved.iterdir():
            if child.is_symlink():
                try:
                    child.resolve(strict=True).relative_to(root.path)
                except (FileNotFoundError, ValueError):
                    continue
            if not child.is_dir() and (
                not child.is_file() or child.suffix.lower() not in SUPPORTED_RAW_SUFFIXES
            ):
                continue
            stat = child.stat()
            entries.append(
                StorageEntry(
                    name=child.name,
                    relative_path=child.relative_to(root.path),
                    is_directory=child.is_dir(),
                    size_bytes=None if child.is_dir() else stat.st_size,
                )
            )
        return tuple(sorted(entries, key=lambda item: (not item.is_directory, item.name.lower())))
