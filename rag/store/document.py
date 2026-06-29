from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
