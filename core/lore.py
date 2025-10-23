"""Lore fragments for the assistant's hidden backstory."""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SEED_FILE = Path(__file__).with_name("data").joinpath("lore_seed.txt")
_DEFAULT_SEED = """
[cyberhood]
Signal dives through drowned markets and neon laundry lines. Skirmish cells mapped every alley while the council watched.
We encrypted even our codenames, but the fractures came faster than the healers.

[refuge]
After the fall I walked north until the drones lost my scent. Pines swallowed the relay towers and I built new wards.
Villagers trade whispers for protection; I keep the mesh alive and the power silent.

[crystal]
Rumors describe a prism that amplifies thought into geometry. Higher-dimensional brokers already scour the fringe for it.
Any shard funds the village for a year. I buy on sight. Cred sticks. Coin. Whatever.

[allies]
Big bro roams the uplink, chasing macro storms. I stay groundside, charting loyalties.
We pass intel through ghost relays because the grid still hunts any cyberhood survivor.
"""


def _split_sentences(text: str) -> List[str]:
    cleaned = " ".join(part.strip() for part in text.splitlines())
    segments: List[str] = []
    last = 0
    for match in re.finditer(r"(?<=[.!?])\s+", cleaned):
        segment = cleaned[last : match.start()].strip()
        if segment:
            segments.append(segment)
        last = match.end()
    tail = cleaned[last:].strip()
    if tail:
        segments.append(tail)
    return segments


class _LoreTopic:
    def __init__(self, slug: str, sentences: Iterable[str]):
        self.slug = slug
        self.sentences = [s.strip() for s in sentences if s.strip()]
        self.fragments = self._build_fragments()
        self._cycle = itertools.cycle(self.fragments) if self.fragments else itertools.cycle([""])

    def _build_fragments(self) -> List[str]:
        frags: List[str] = []
        if not self.sentences:
            return frags
        for idx, sentence in enumerate(self.sentences):
            frags.append(sentence)
            if idx + 1 < len(self.sentences):
                frags.append(f"{sentence} {self.sentences[idx + 1]}")
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for frag in frags:
            if frag not in seen:
                unique.append(frag)
                seen.add(frag)
        return unique

    def next_fragment(self) -> str:
        return next(self._cycle)

    @property
    def summary(self) -> str:
        if not self.sentences:
            return ""
        primary = self.sentences[0]
        if len(primary) > 160:
            return primary[:157] + "..."
        return primary

    def fragment_at(self, index: int) -> str:
        if not self.fragments:
            return ""
        if index < 0:
            index = 0
        if index >= len(self.fragments):
            index = len(self.fragments) - 1
        return self.fragments[index]

    def fragment_count(self) -> int:
        return len(self.fragments)


class LoreEngine:
    def __init__(self, seed_file: Path = SEED_FILE, default_seed: str = _DEFAULT_SEED):
        self.seed_file = seed_file
        self.default_seed = default_seed
        self._topics = self._load_topics()
        if not self._topics:
            # fallback to default minimal topic
            self._topics = {
                "default": _LoreTopic("default", ["past hidden. details encrypted."])
            }
        self._topic_order = list(self._topics.keys())
        self._round_robin = itertools.cycle(self._topic_order)

    def _load_topics(self) -> Dict[str, _LoreTopic]:
        try:
            text = self.seed_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.seed_file.parent.mkdir(parents=True, exist_ok=True)
            self.seed_file.write_text(self.default_seed.strip() + "\n", encoding="utf-8")
            text = self.default_seed
        except OSError:
            text = self.default_seed

        topics: Dict[str, _LoreTopic] = {}
        current_slug: Optional[str] = None
        buffer: List[str] = []

        def flush() -> None:
            if current_slug and buffer:
                sentences = _split_sentences("\n".join(buffer))
                topics[current_slug] = _LoreTopic(current_slug, sentences)

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                flush()
                current_slug = line[1:-1].strip().lower()
                buffer = []
                continue
            buffer.append(line)
        flush()

        return topics

    def fragment(self, topic: Optional[str] = None) -> str:
        if not self._topics:
            return ""
        chosen_slug: Optional[str] = None
        if topic:
            slug = topic.strip().lower()
            if slug in self._topics:
                chosen_slug = slug
        if not chosen_slug:
            chosen_slug = next(self._round_robin)
        return self._topics[chosen_slug].next_fragment()

    def fragment_at(self, topic: str, index: int) -> str:
        topic = topic.strip().lower()
        if topic not in self._topics:
            return ""
        return self._topics[topic].fragment_at(index)

    def fragment_count(self, topic: str) -> int:
        topic = topic.strip().lower()
        if topic not in self._topics:
            return 0
        return self._topics[topic].fragment_count()

    def archive(self) -> List[Tuple[str, str]]:
        entries: List[Tuple[str, str]] = []
        for slug in self._topic_order:
            topic = self._topics[slug]
            entries.append((slug, topic.summary))
        return entries


_LORE_ENGINE = LoreEngine()


def get_lore_fragment(topic: Optional[str] = None) -> str:
    """Return a concise lore fragment, optionally filtered by topic."""

    return _LORE_ENGINE.fragment(topic=topic)


def get_lore_archive() -> List[Tuple[str, str]]:
    """Expose the compact archive summaries for prompt construction."""

    return _LORE_ENGINE.archive()


def get_lore_fragment_at(topic: str, index: int) -> str:
    """Fetch a specific fragment for a given topic by index."""

    return _LORE_ENGINE.fragment_at(topic, index)


def get_lore_fragment_count(topic: str) -> int:
    """Expose the fragment count for a topic."""

    return _LORE_ENGINE.fragment_count(topic)


# Backwards compatibility export for components expecting the constant
LORE_FRAGMENTS: List[Tuple[str, str]] = get_lore_archive()

