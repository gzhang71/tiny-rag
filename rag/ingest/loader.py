from pathlib import Path


def load_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_directory(directory: str | Path, glob: str = "**/*.txt") -> dict[str, str]:
    root = Path(directory)
    return {str(p): p.read_text(encoding="utf-8") for p in root.glob(glob)}
