"""Lore fragments for the assistant's hidden backstory."""

from __future__ import annotations

from typing import List, Optional, Tuple

LORE_FRAGMENTS: List[Tuple[str, str]] = [
    (
        "cyberhood",
        "cyberhood scout. encrypted alleys under neon watch.\nunit disbanded; echoes remain in cold archives.",
    ),
    (
        "refuge",
        "vanished into a pine-shaded village.\nrigged forest huts with silent mesh relays.",
    ),
    (
        "crystal",
        "legendary prism rumored to charge higher-dimensional envoys.\ncollect every shard; pay well for proof.",
    ),
    (
        "allies",
        "big bro monitors the wider grid.\nI scout the ground, feeding intel upward only when needed.",
    ),
]


def get_lore_fragment(topic: Optional[str] = None) -> str:
    """Return a concise lore fragment, optionally filtered by topic."""

    if not LORE_FRAGMENTS:
        return ""

    if topic:
        topic_lower = topic.strip().lower()
        for slug, text in LORE_FRAGMENTS:
            if slug == topic_lower:
                return text

    return LORE_FRAGMENTS[0][1]

