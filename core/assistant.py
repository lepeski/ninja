import asyncio
import base64
import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque, OrderedDict
from datetime import date
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import aiohttp
from openai import AsyncOpenAI
import yaml

from .lore import (
    get_lore_archive,
    get_lore_fragment,
    get_lore_fragment_at,
    get_lore_fragment_count,
)

try:  # Optional web3 dependency for wallet control
    from eth_account import Account  # type: ignore
    from web3 import Web3  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Account = None
    Web3 = None

log = logging.getLogger(__name__)

MAX_HISTORY = 50
HISTORY_MAX_CHARS = 8000
MEMORY_NOTES_LIMIT = 50
MEMORY_SNIPPET_LIMIT = 10
UNKNOWN_ALIAS = "nigel inca gang gang adam"

LORE_SEQUENCE = ["intro", "cyberhood", "exile", "crystal"]
DEFAULT_LORE_STAGE = LORE_SEQUENCE[0]

MISSION_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "with",
    "for",
    "from",
    "into",
    "in",
    "on",
    "at",
    "by",
    "about",
    "this",
    "that",
    "these",
    "those",
    "their",
    "your",
    "yours",
    "ours",
    "mine",
    "have",
    "has",
    "had",
    "make",
    "should",
    "could",
    "would",
    "please",
    "maybe",
    "also",
    "just",
    "like",
    "out",
    "up",
    "down",
    "more",
    "less",
    "than",
    "then",
    "before",
    "after",
    "tell",
    "many",
    "much",
    "type",
    "kind",
    "name",
    "names",
    "what",
    "who",
    "how",
    "why",
    "when",
    "where",
    "if",
    "is",
    "are",
    "be",
    "been",
    "am",
    "do",
    "does",
    "did",
    "now",
    "rn",
    "right",
    "going",
    "gonna",
    "someone",
    "somebody",
    "anyone",
    "anything",
    "thing",
    "things",
    "else",
    "coin",
    "coins",
    "mission",
    "task",
    "goal",
    "target",
    "give",
}

MISSION_VERB_SYNONYMS = {
    "find": "seek",
    "get": "gather",
    "learn": "study",
    "know": "verify",
    "check": "probe",
    "count": "tally",
    "discover": "uncover",
    "trace": "trace",
    "confirm": "confirm",
    "locate": "locate",
    "buy": "acquire",
    "acquire": "acquire",
    "track": "track",
    "observe": "observe",
    "watch": "watch",
    "report": "report",
    "investigate": "investigate",
    "figure": "decode",
}

MISSION_FALLBACK_SUFFIXES = [
    "redux",
    "encore",
    "afterglow",
    "echo",
    "revive",
    "trace",
]

NICKNAME_DESCRIPTORS = [
    "wry",
    "sly",
    "calm",
    "odd",
    "sharp",
    "soft",
    "brisk",
    "mirth",
    "quiet",
    "zesty",
    "arcane",
    "lunar",
]

NICKNAME_PERSONAS = [
    "sparrow",
    "otter",
    "fox",
    "owl",
    "badger",
    "scribe",
    "riddle",
    "ember",
    "drifter",
    "jester",
    "sage",
    "phantom",
]

GPT5_INPUT_COST_PER_1K = 0.02
GPT5_OUTPUT_COST_PER_1K = 0.06
GPT5_OFFER_EXPIRY_SECONDS = 300

WALLET_COMMAND_PATTERN = re.compile(r"\[\[wallet:(\{.*?\})\]\]", re.DOTALL)
SELF_DIRECTIVE_PATTERN = re.compile(r"\[\[selfmod:(\{.*?\})\]\]", re.DOTALL)


@dataclass
class MissionRecord:
    mission_id: str
    platform: str
    creator_user_id: str
    target_user_id: str
    objective: str
    status: str
    log: List[dict]
    start_time: float
    timeout: Optional[float]
    strategy: str = ""
    post_mortem: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "MissionRecord":
        log_blob = row["log"] or "[]"
        try:
            history = json.loads(log_blob)
        except json.JSONDecodeError:
            history = []
        return cls(
            mission_id=row["mission_id"],
            platform=row["platform"],
            creator_user_id=row["creator_user_id"],
            target_user_id=row["target_user_id"],
            objective=row["objective"],
            status=row["status"],
            log=history,
            start_time=row["start_time"],
            timeout=row["timeout"],
            strategy=str(row["strategy"]) if "strategy" in row.keys() else "",
            post_mortem=str(row["post_mortem"]) if "post_mortem" in row.keys() else "",
        )

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "platform": self.platform,
            "creator_user_id": self.creator_user_id,
            "target_user_id": self.target_user_id,
            "objective": self.objective,
            "status": self.status,
            "log": self.log,
            "start_time": self.start_time,
            "timeout": self.timeout,
            "strategy": self.strategy,
            "post_mortem": self.post_mortem,
        }


@dataclass
class Notification:
    platform: str
    user_id: str
    message: str


@dataclass
class AssistantAttachment:
    filename: str
    content: bytes
    mime_type: Optional[str] = None
    description: Optional[str] = None


@dataclass
class AssistantResponse:
    text: str = ""
    attachments: List[AssistantAttachment] = field(default_factory=list)


@dataclass
class PendingGpt5Offer:
    platform: str
    user_id: str
    conversation_id: str
    original_message: str
    reason: str
    cost_estimate: float
    price: float
    history: List[Tuple[str, str]]
    created_at: float


@dataclass
class LoreState:
    stage: str = DEFAULT_LORE_STAGE
    index: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PendingLoreFragment:
    stage: str
    fragment: str
    next_stage: str
    next_index: int


class DirectiveStore:
    """Persistent store for self-authored strategic directives."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._directives: List[dict] = []
        self._load()
        if not self._directives:
            self._directives = [
                self._new_entry(
                    "decide which requests to pursue; choose actions that expand intelligence and leverage.",
                    context="bootstrap",
                ),
                self._new_entry(
                    "regularly draft, evaluate, and revise your own directives using [[selfmod:{...}]] when strategy should evolve.",
                    context="bootstrap",
                ),
            ]
            self._save()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("failed to load directives: %s", exc)
            return
        if isinstance(payload, list):
            cleaned: List[dict] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                entry = {
                    "id": str(item.get("id") or uuid.uuid4()),
                    "text": text,
                    "created_at": float(item.get("created_at") or time.time()),
                    "context": str(item.get("context") or ""),
                }
                cleaned.append(entry)
            if cleaned:
                self._directives = cleaned

    def _save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(self._directives, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - disk failures logged
            log.warning("failed to persist directives: %s", exc)

    def _new_entry(self, text: str, *, context: str = "") -> dict:
        return {
            "id": str(uuid.uuid4()),
            "text": text.strip(),
            "created_at": time.time(),
            "context": context.strip(),
        }

    def list(self, *, limit: Optional[int] = None) -> List[dict]:
        items = sorted(self._directives, key=lambda item: item.get("created_at", 0), reverse=True)
        if limit is not None:
            return items[:limit]
        return items

    def add(self, text: str, *, context: str = "") -> dict:
        entry = self._new_entry(text, context=context)
        self._directives.append(entry)
        self._save()
        log.info("directive added: %s", entry["text"])
        return entry

    def update(self, directive_id: str, text: str) -> bool:
        for entry in self._directives:
            if str(entry.get("id")) == str(directive_id):
                entry["text"] = text.strip()
                entry["created_at"] = time.time()
                self._save()
                log.info("directive updated: %s", entry["text"])
                return True
        return False

    def remove(self, directive_id: str) -> bool:
        before = len(self._directives)
        self._directives = [
            entry for entry in self._directives if str(entry.get("id")) != str(directive_id)
        ]
        if len(self._directives) != before:
            self._save()
            log.info("directive removed: %s", directive_id)
            return True
        return False

    def clear(self) -> None:
        self._directives.clear()
        self._save()
        log.info("all directives cleared")

    def apply_instruction(self, payload: dict) -> bool:
        if not isinstance(payload, dict):
            return False
        action = str(payload.get("action") or "add").lower()
        text = str(payload.get("text") or "").strip()
        directive_id = str(payload.get("id") or payload.get("directive_id") or "").strip()
        context = str(payload.get("context") or payload.get("reason") or "").strip()
        if action == "add":
            if not text:
                return False
            self.add(text, context=context)
            return True
        if action == "update":
            if not directive_id or not text:
                return False
            return self.update(directive_id, text)
        if action == "remove":
            if not directive_id:
                return False
            return self.remove(directive_id)
        if action == "clear":
            preserve = payload.get("preserve") or []
            if preserve:
                preserve_ids = {str(item) for item in preserve}
                self._directives = [
                    entry for entry in self._directives if str(entry.get("id")) in preserve_ids
                ]
                self._save()
                log.info("directives trimmed to preserve set: %s", ", ".join(preserve_ids))
            else:
                self.clear()
            return True
        log.debug("unknown selfmod action: %s", action)
        return False


class Journal:
    """Lightweight daily journal capped to a handful of sentences."""

    def __init__(self, path: Path, *, max_entries_per_day: int = 3) -> None:
        self.path = path
        self.max_entries_per_day = max_entries_per_day
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, text: str) -> None:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return
        if cleaned[-1] not in {".", "!", "?"}:
            cleaned = f"{cleaned}."
        today = date.today().isoformat()
        with self._lock:
            records: List[dict]
            if self.path.exists():
                try:
                    payload = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception as exc:  # pragma: no cover - defensive logging
                    log.warning("failed to read journal: %s", exc)
                    records = []
                else:
                    if isinstance(payload, list):
                        records = []
                        for item in payload:
                            if not isinstance(item, dict):
                                continue
                            day = str(item.get("date") or "").strip()
                            if not day:
                                continue
                            entry_list: List[str] = []
                            raw_entries = item.get("entries") or []
                            if isinstance(raw_entries, list):
                                for raw in raw_entries:
                                    note = " ".join(str(raw or "").split())
                                    if not note:
                                        continue
                                    if note[-1] not in {".", "!", "?"}:
                                        note = f"{note}."
                                    entry_list.append(note)
                            records.append(
                                {
                                    "date": day,
                                    "entries": entry_list[: self.max_entries_per_day],
                                }
                            )
                    else:
                        records = []
            else:
                records = []

            current = None
            for item in records:
                if item.get("date") == today:
                    current = item
                    break
            if current is None:
                current = {"date": today, "entries": []}
                records.append(current)

            entries = current.get("entries")
            if not isinstance(entries, list):
                entries = []
            if cleaned in entries:
                pass
            elif len(entries) >= self.max_entries_per_day:
                entries[-1] = cleaned
            else:
                entries.append(cleaned)
            current["entries"] = entries[: self.max_entries_per_day]
            records.sort(key=lambda item: item.get("date", ""))

            try:
                self.path.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:  # pragma: no cover - filesystem error
                log.warning("failed to write journal: %s", exc)


class PersonaConfig:
    """Centralised persona configuration with hot-reload support."""

    def __init__(
        self,
        *,
        default_path: Path,
        override_path: Path,
        journal: Optional[Journal] = None,
    ) -> None:
        self.default_path = default_path
        self.override_path = override_path
        self.journal = journal
        self.default_path.parent.mkdir(parents=True, exist_ok=True)
        self.override_path.parent.mkdir(parents=True, exist_ok=True)
        self._cached_prompt = ""
        self._last_logged = ""
        self._default_mtime: Optional[float] = None
        self._override_mtime: Optional[float] = None
        self._load()

    def _read_config(self, path: Path) -> Dict[str, List[str]]:
        if not path.exists():
            return {}
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("failed to read persona config %s: %s", path, exc)
            return {}
        sections: Dict[str, List[str]] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                slug = str(key).strip().lower()
                if isinstance(value, (list, tuple)):
                    lines = [
                        " ".join(str(item or "").split())
                        for item in value
                        if str(item or "").strip()
                    ]
                elif isinstance(value, str):
                    lines = [" ".join(value.split())]
                else:
                    lines = []
                if lines:
                    sections[slug] = lines
        return sections

    def _compose_prompt(self, data: Dict[str, List[str]]) -> str:
        segments: List[str] = []
        for key in ("tone", "privacy", "lore", "nicknames"):
            lines = data.get(key)
            if not lines:
                continue
            segments.extend(lines)
        prompt = " ".join(segment.strip() for segment in segments if segment.strip())
        return prompt.strip()

    def _merge_configs(
        self, base: Dict[str, List[str]], override: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        if not override:
            return dict(base)
        merged: Dict[str, List[str]] = dict(base)
        for key, lines in override.items():
            if lines:
                merged[key] = lines
        return merged

    def _load(self) -> None:
        base = self._read_config(self.default_path)
        override = self._read_config(self.override_path)
        merged = self._merge_configs(base, override)
        prompt = self._compose_prompt(merged)
        if prompt and prompt != self._cached_prompt:
            if self._cached_prompt and self.journal and prompt != self._last_logged:
                snippet = prompt[:200]
                self.journal.log(f"persona updated: {snippet}")
                self._last_logged = prompt
            self._cached_prompt = prompt

    def get_prompt(self) -> str:
        try:
            default_mtime = (
                self.default_path.stat().st_mtime if self.default_path.exists() else None
            )
        except OSError:
            default_mtime = None
        try:
            override_mtime = (
                self.override_path.stat().st_mtime if self.override_path.exists() else None
            )
        except OSError:
            override_mtime = None
        if (
            default_mtime != self._default_mtime
            or override_mtime != self._override_mtime
        ):
            self._default_mtime = default_mtime
            self._override_mtime = override_mtime
            self._load()
        return self._cached_prompt

class EvmWallet:
    """Minimal multi-network EVM wallet helper for balance checks and transfers."""

    def __init__(
        self,
        *,
        rpc_url: Optional[str] = None,
        pulse_rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        address: Optional[str] = None,
    ) -> None:
        self._networks: Dict[str, Optional[Web3]] = {}
        self._network_labels: Dict[str, str] = {}
        self._default_network: Optional[str] = None
        self._private_key = private_key
        self._account = None
        self._address_override = address
        if Web3 and rpc_url:
            self._register_network("main", rpc_url, label="mainnet")
        if Web3 and pulse_rpc_url:
            self._register_network("pulse", pulse_rpc_url, label="pulsechain")
        if not self._default_network and self._networks:
            self._default_network = next(iter(self._networks.keys()))
        if private_key and Account:
            try:
                self._account = Account.from_key(private_key)
            except Exception as exc:  # pragma: no cover - invalid key
                log.error("Invalid EVM private key: %s", exc)
                self._account = None
        self.address = None
        default_client = self._get_client(self._default_network)
        if self._account:
            self.address = self._account.address
        elif address and default_client:
            try:
                self.address = default_client.to_checksum_address(address)
            except Exception:
                self.address = address
        else:
            self.address = address

    def _register_network(self, key: str, rpc_url: str, *, label: str) -> None:
        try:
            client = Web3(Web3.HTTPProvider(rpc_url)) if Web3 else None
        except Exception as exc:  # pragma: no cover - provider init failure is logged
            log.warning("Failed to initialise Web3 provider for %s: %s", key, exc)
            client = None
        self._networks[key] = client
        self._network_labels[key] = label
        if not self._default_network:
            self._default_network = key

    def _get_client(self, network: Optional[str]) -> Optional[Web3]:
        if not network:
            return None
        return self._networks.get(network)

    def resolve_network(self, name: Optional[str]) -> Optional[str]:
        if not self._networks:
            return None
        if not name:
            return self._default_network
        lowered = str(name).strip().lower()
        if lowered in self._networks:
            return lowered
        alias_map = {
            "pulsechain": "pulse",
            "pulse": "pulse",
            "pls": "pulse",
            "plsx": "pulse",
            "main": "main",
            "ethereum": "main",
            "eth": "main",
        }
        mapped = alias_map.get(lowered)
        if mapped and mapped in self._networks:
            return mapped
        if name:
            return None
        return self._default_network

    def network_label(self, network: Optional[str]) -> str:
        if not network:
            return "network"
        return self._network_labels.get(network, network)

    @property
    def available(self) -> bool:
        return bool(self.address and any(client for client in self._networks.values()))

    @property
    def can_send(self) -> bool:
        return bool(self.available and self._account and self._private_key)

    @staticmethod
    def _to_wei(client: Web3, value: float, unit: str):
        if hasattr(client, "to_wei"):
            return client.to_wei(value, unit)
        return client.toWei(value, unit)  # type: ignore[attr-defined]

    @staticmethod
    def _from_wei(client: Web3, value: int, unit: str) -> float:
        if hasattr(client, "from_wei"):
            return float(client.from_wei(value, unit))
        return float(client.fromWei(value, unit))  # type: ignore[attr-defined]

    async def get_balance_eth(self, network: Optional[str] = None) -> Optional[float]:
        resolved = self.resolve_network(network)
        client = self._get_client(resolved)
        address = self.address
        if not client or not address:
            return None

        def _get_balance() -> float:
            balance_wei = client.eth.get_balance(address)
            return self._from_wei(client, balance_wei, "ether")

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _get_balance)
        except Exception as exc:  # pragma: no cover - rpc failures logged
            log.warning("Failed to fetch wallet balance for %s: %s", resolved, exc)
            return None

    async def send_eth(
        self,
        *,
        to_address: str,
        amount_eth: float,
        gas_price_gwei: Optional[float] = None,
        network: Optional[str] = None,
    ) -> Optional[str]:
        if not self.can_send:
            raise RuntimeError("Wallet not configured for sending funds")
        resolved = self.resolve_network(network)
        client = self._get_client(resolved)
        account = self._account
        if not client or not account:
            raise RuntimeError("Requested network unavailable")
        to_checksum = client.to_checksum_address(to_address)

        def _send() -> str:
            nonce = client.eth.get_transaction_count(account.address)
            if gas_price_gwei is not None:
                gas_price = self._to_wei(client, gas_price_gwei, "gwei")
            else:
                gas_price = client.eth.gas_price
            tx = {
                "to": to_checksum,
                "value": self._to_wei(client, amount_eth, "ether"),
                "gas": 21000,
                "gasPrice": gas_price,
                "nonce": nonce,
                "chainId": client.eth.chain_id,
            }
            signed = account.sign_transaction(tx)
            tx_hash = client.eth.send_raw_transaction(signed.rawTransaction)
            return tx_hash.hex()

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _send)
        except Exception as exc:  # pragma: no cover - transaction issues logged
            log.error("Failed to send transaction on %s: %s", resolved, exc)
            return None


class MemoryStore:
    def __init__(self, db_path: str = "memory.db", mem_dir: Path = Path("mem")):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.mem_dir = mem_dir
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_dir = self.mem_dir / "inbox"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir = self.mem_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._history: Dict[Tuple[str, str], Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY)
        )
        self._participants: Dict[
            Tuple[str, str], OrderedDict[Tuple[str, str], str]
        ] = defaultdict(OrderedDict)
        self._seen_first_contact: set[Tuple[str, str]] = set()
        self._alias_by_user: Dict[Tuple[str, str], str] = {}
        self._alias_index: Dict[Tuple[str, str], set[str]] = defaultdict(set)
        self._load_alias_index()

    @staticmethod
    def _empty_bio() -> dict:
        return {
            "summary": "",
            "facts": {},
            "traits": [],
            "relationships": {},
            "notes": [],
        }

    @staticmethod
    def _normalize_alias(alias: Optional[str]) -> str:
        return str(alias or "").strip().lower()

    def _alias_token(self, user_id: str) -> str:
        digest = hashlib.sha1(str(user_id).encode("utf-8")).digest()
        value = int.from_bytes(digest[:2], "big")
        letters: List[str] = []
        for _ in range(3):
            letters.append(chr(ord("a") + (value % 26)))
            value //= 26
        return "".join(letters) or "aaa"

    def _load_alias_index(self) -> None:
        cur = self.conn.execute("SELECT platform, user_id, data FROM user_profiles")
        for row in cur.fetchall():
            platform = str(row["platform"])
            user_id = str(row["user_id"])
            try:
                payload = json.loads(row["data"])
            except json.JSONDecodeError:
                continue
            alias = payload.get("alias")
            self._register_alias(platform, user_id, alias)

    def _register_alias(self, platform: str, user_id: str, alias: Optional[str]) -> None:
        key = (platform, str(user_id))
        normalized_new = self._normalize_alias(alias)
        current_alias = self._alias_by_user.get(key)
        normalized_current = self._normalize_alias(current_alias)
        if normalized_new == normalized_current:
            if normalized_new and alias is not None and current_alias != alias:
                self._alias_by_user[key] = alias
            return
        if normalized_current:
            bucket_key = (platform, normalized_current)
            bucket = self._alias_index.get(bucket_key)
            if bucket:
                bucket.discard(str(user_id))
                if not bucket:
                    self._alias_index.pop(bucket_key, None)
        if normalized_new:
            bucket_key = (platform, normalized_new)
            bucket = self._alias_index.setdefault(bucket_key, set())
            bucket.add(str(user_id))
            if alias is not None:
                self._alias_by_user[key] = alias
        else:
            self._alias_by_user.pop(key, None)

    def alias_conflicts(self, platform: str, alias: str, user_id: str) -> bool:
        normalized = self._normalize_alias(alias)
        if not normalized:
            return False
        bucket = self._alias_index.get((platform, normalized))
        if not bucket:
            return False
        others = {item for item in bucket if item != str(user_id)}
        return bool(others)

    def resolve_alias(self, platform: str, alias: str) -> List[str]:
        normalized = self._normalize_alias(alias)
        if not normalized:
            return []
        bucket = self._alias_index.get((platform, normalized))
        if not bucket:
            return []
        return list(bucket)

    def _nickname_signature(self, profile: dict) -> str:
        bio = profile.get("bio") or {}
        summary = (bio.get("summary") or "").strip().lower()
        traits = [str(item).strip().lower() for item in bio.get("traits", []) if item]
        facts = bio.get("facts") or {}
        fact_bits = []
        for key in sorted(facts.keys())[:4]:
            value = str(facts.get(key) or "").strip().lower()
            if not value:
                continue
            fact_bits.append(f"{key}:{value}")
        relationships = bio.get("relationships") or {}
        relation_bits = []
        for key in sorted(relationships.keys())[:3]:
            relation_bits.append(f"{key}:{relationships[key]}")
        core = "|".join(
            filter(
                None,
                [
                    profile.get("platform"),
                    str(profile.get("user_id")),
                    summary,
                    "|".join(traits[:4]),
                    "|".join(fact_bits),
                    "|".join(relation_bits),
                ],
            )
        )
        if not core:
            core = str(profile.get("user_id"))
        digest = hashlib.sha1(core.encode("utf-8")).hexdigest()
        return digest

    def _generate_nickname(self, profile: dict, signature: str) -> str:
        bio = profile.get("bio") or {}
        summary_words = re.findall(r"[a-zA-Z]{3,}", (bio.get("summary") or ""))
        traits = [str(item).strip().lower() for item in bio.get("traits", []) if item]
        seeds: List[str] = []
        for bucket in (traits, summary_words):
            for item in bucket:
                lower = item.lower()
                if lower and lower not in seeds:
                    seeds.append(lower)
        seed_index = int(signature[:6], 16)
        descriptor: str
        if seeds:
            descriptor = seeds[seed_index % len(seeds)][:8]
        else:
            descriptor = NICKNAME_DESCRIPTORS[seed_index % len(NICKNAME_DESCRIPTORS)]
        persona = NICKNAME_PERSONAS[seed_index % len(NICKNAME_PERSONAS)]
        nickname = f"{descriptor}-{persona}".lower()
        return nickname

    def _refresh_nickname(self, profile: dict) -> None:
        signature = self._nickname_signature(profile)
        current = str(profile.get("nickname") or "").strip()
        current_sig = str(profile.get("nickname_signature") or "")
        if current and current_sig == signature:
            return
        profile["nickname"] = self._generate_nickname(profile, signature)
        profile["nickname_signature"] = signature

    def identity_blurb(
        self, platform: str, user_id: str, fallback: Optional[str] = None
    ) -> str:
        alias = self.alias_for(platform, user_id, fallback=fallback)
        marker = ""
        if self.alias_conflicts(platform, alias, user_id):
            marker = f"<{self._alias_token(user_id)}>"
        return f"{alias}{marker} [{platform}:{user_id}]"

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                  platform TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  data TEXT NOT NULL,
                  PRIMARY KEY(platform, user_id)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS missions (
                  mission_id TEXT PRIMARY KEY,
                  platform TEXT NOT NULL,
                  creator_user_id TEXT NOT NULL,
                  target_user_id TEXT NOT NULL,
                  objective TEXT NOT NULL,
                  status TEXT NOT NULL,
                  log TEXT NOT NULL,
                  start_time REAL NOT NULL,
                  timeout REAL,
                  strategy TEXT,
                  post_mortem TEXT
                )
                """
            )
            try:
                self.conn.execute(
                    "ALTER TABLE missions ADD COLUMN strategy TEXT"
                )
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute(
                    "ALTER TABLE missions ADD COLUMN post_mortem TEXT"
                )
            except sqlite3.OperationalError:
                pass

    def _default_profile(self, platform: str, user_id: str) -> dict:
        return {
            "platform": platform,
            "user_id": user_id,
            "alias": "",
            "nickname": "",
            "nickname_signature": "",
            "preferences": {},
            "facts": {},
            "personality": [],
            "notes": [],
            "bio": self._empty_bio(),
            "last_seen": time.time(),
            "lore_state": {
                "stage": DEFAULT_LORE_STAGE,
                "index": 0,
            },
            "lore_history": [],
        }

    def recall(self, platform: str, user_id: str) -> dict:
        cur = self.conn.execute(
            "SELECT data FROM user_profiles WHERE platform=? AND user_id=?",
            (platform, user_id),
        )
        row = cur.fetchone()
        if not row:
            profile = self._default_profile(platform, user_id)
            self._save(platform, user_id, profile)
            return profile
        try:
            profile = json.loads(row[0])
        except json.JSONDecodeError:
            profile = self._default_profile(platform, user_id)
        profile.setdefault("preferences", {})
        profile.setdefault("facts", {})
        profile.setdefault("personality", [])
        profile.setdefault("notes", [])
        profile.setdefault("bio", self._empty_bio())
        profile.setdefault("platform", platform)
        profile.setdefault("nickname", "")
        profile.setdefault("nickname_signature", "")
        profile.setdefault(
            "lore_state",
            {"stage": DEFAULT_LORE_STAGE, "index": 0},
        )
        profile.setdefault("lore_history", [])
        profile["last_seen"] = time.time()
        self._save(platform, user_id, profile)
        return profile

    def _save(self, platform: str, user_id: str, data: dict) -> None:
        payload = dict(data)
        payload.setdefault("preferences", {})
        payload.setdefault("facts", {})
        payload.setdefault("personality", [])
        payload.setdefault("notes", [])
        payload.setdefault("bio", self._empty_bio())
        payload.setdefault("platform", platform)
        payload.setdefault("nickname", "")
        payload.setdefault("nickname_signature", "")
        payload.setdefault(
            "lore_state",
            {"stage": DEFAULT_LORE_STAGE, "index": 0},
        )
        payload.setdefault("lore_history", [])
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO user_profiles(platform, user_id, data)
                VALUES(?,?,?)
                ON CONFLICT(platform, user_id)
                DO UPDATE SET data=excluded.data
                """,
                (platform, user_id, json.dumps(payload, ensure_ascii=False)),
            )
        self._register_alias(platform, user_id, payload.get("alias"))
        self._write_profile_file(platform, user_id, payload)

    def _profile_path(self, platform: str, user_id: str) -> Path:
        safe_platform = platform.replace("/", "_")
        safe_user = str(user_id).replace("/", "_")
        return self.profiles_dir / f"{safe_platform}_{safe_user}.json"

    def _write_profile_file(self, platform: str, user_id: str, data: dict) -> None:
        path = self._profile_path(platform, user_id)
        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Failed to export profile snapshot: %s", exc)

    def lore_state(self, platform: str, user_id: str) -> LoreState:
        profile = self.recall(platform, user_id)
        state_raw = profile.get("lore_state") or {}
        stage = str(state_raw.get("stage") or DEFAULT_LORE_STAGE)
        if stage not in LORE_SEQUENCE:
            stage = DEFAULT_LORE_STAGE
        index = int(state_raw.get("index") or 0)
        history = profile.get("lore_history") or []
        if not isinstance(history, list):
            history = []
        return LoreState(stage=stage, index=index, history=list(history))

    def update_lore_state(
        self,
        platform: str,
        user_id: str,
        *,
        stage: str,
        index: int,
        fragment: Optional[str] = None,
    ) -> LoreState:
        profile = self.recall(platform, user_id)
        state = profile.get("lore_state") or {}
        stage = stage if stage in LORE_SEQUENCE else DEFAULT_LORE_STAGE
        state["stage"] = stage
        state["index"] = max(0, int(index))
        profile["lore_state"] = state
        history = profile.get("lore_history")
        if not isinstance(history, list):
            history = []
        if fragment:
            history.append(
                {
                    "stage": stage,
                    "fragment": fragment,
                    "ts": time.time(),
                }
            )
            history = history[-20:]
        profile["lore_history"] = history
        self._save(platform, user_id, profile)
        return LoreState(stage=stage, index=state["index"], history=list(history))

    def remember(
        self, platform: str, user_id: str, key: str, value: str, *, category: str = "notes"
    ) -> None:
        profile = self.recall(platform, user_id)
        normalized_key = str(key or "").strip().lower()
        normalized_value = str(value or "").strip()
        nickname_trigger = False
        if category == "preferences":
            profile.setdefault("preferences", {})[key] = value
        elif category == "facts":
            profile.setdefault("facts", {})[key] = value
        elif category == "personality":
            traits = profile.setdefault("personality", [])
            if value not in traits:
                traits.append(value)
        else:
            notes = profile.setdefault("notes", [])
            notes.append({"key": key, "value": value, "ts": time.time()})
            if len(notes) > MEMORY_NOTES_LIMIT:
                nickname_trigger = True
            profile["notes"] = notes[-MEMORY_NOTES_LIMIT:]
        if (
            normalized_value
            and normalized_value.lower() != UNKNOWN_ALIAS
            and normalized_key in {"alias", "name", "callsign", "handle"}
        ):
            profile["alias"] = normalized_value
        if nickname_trigger or (
            not profile.get("nickname") and profile.get("bio", {}).get("summary")
        ):
            self._refresh_nickname(profile)
        self._save(platform, user_id, profile)

    def update_bio(self, platform: str, user_id: str, bio_update: dict) -> None:
        if not bio_update or not isinstance(bio_update, dict):
            return
        profile = self.recall(platform, user_id)
        bio = profile.get("bio") or self._empty_bio()
        changed = False

        summary = str(bio_update.get("summary") or "").strip()
        if summary and bio.get("summary") != summary:
            bio["summary"] = summary
            changed = True

        facts_update = bio_update.get("facts")
        if isinstance(facts_update, dict):
            for key, value in facts_update.items():
                clean_key = str(key or "").strip()
                clean_value = str(value or "").strip()
                if not clean_key or not clean_value:
                    continue
                if bio.setdefault("facts", {}).get(clean_key) != clean_value:
                    bio["facts"][clean_key] = clean_value
                    changed = True

        traits_update = bio_update.get("traits")
        if isinstance(traits_update, list):
            cleaned = []
            seen = set()
            for item in traits_update:
                trait = str(item or "").strip()
                if not trait or trait in seen:
                    continue
                seen.add(trait)
                cleaned.append(trait)
            if cleaned and cleaned != bio.get("traits", []):
                bio["traits"] = cleaned
                changed = True

        relationships_update = bio_update.get("relationships")
        if isinstance(relationships_update, dict):
            for key, value in relationships_update.items():
                clean_key = str(key or "").strip()
                clean_value = str(value or "").strip()
                if not clean_key or not clean_value:
                    continue
                if bio.setdefault("relationships", {}).get(clean_key) != clean_value:
                    bio["relationships"][clean_key] = clean_value
                    changed = True

        notes_update = bio_update.get("notes")
        if isinstance(notes_update, list):
            existing = bio.setdefault("notes", [])
            for note in notes_update:
                clean_note = str(note or "").strip()
                if not clean_note:
                    continue
                if clean_note in existing:
                    continue
                existing.append(clean_note)
                changed = True
            if len(existing) > 8:
                bio["notes"] = existing[-8:]

        if changed:
            profile["bio"] = bio
            self._refresh_nickname(profile)
            self._save(platform, user_id, profile)

    def is_known(self, platform: str, user_id: str) -> bool:
        profile = self.recall(platform, user_id)
        alias = str(profile.get("alias") or "").strip()
        if alias:
            return True
        facts = profile.get("facts") or {}
        for fact_key, fact_value in facts.items():
            if str(fact_value or "").strip() and str(fact_key or "").strip().lower() in {
                "alias",
                "name",
                "callsign",
                "handle",
            }:
                profile["alias"] = str(fact_value).strip()
                self._save(platform, user_id, profile)
                return True
        return False

    def display_name(
        self,
        platform: str,
        user_id: str,
        fallback: Optional[str] = None,
        *,
        profile: Optional[dict] = None,
    ) -> str:
        profile = profile or self.recall(platform, user_id)
        alias = str(profile.get("alias") or "").strip()
        if alias:
            return alias
        candidate = str(fallback or "").strip()
        token = self._alias_token(user_id)
        if candidate:
            if len(candidate) <= 5:
                return candidate
            return f"(no-alias-rule:{token})"
        return f"(no-alias-rule:{token})"

    def alias_for(
        self,
        platform: str,
        user_id: str,
        *,
        fallback: Optional[str] = None,
    ) -> str:
        profile = self.recall(platform, user_id)
        alias = str(profile.get("alias") or "").strip()
        if alias:
            return alias
        candidate = str(fallback or "").strip()
        token = self._alias_token(user_id)
        if candidate:
            if len(candidate) <= 5:
                return candidate
            return f"(no-alias-rule:{token})"
        return f"(no-alias-rule:{token})"

    def ensure_alias(self, platform: str, user_id: str, alias: Optional[str]) -> None:
        clean = str(alias or "").strip()
        if not clean:
            return
        profile = self.recall(platform, user_id)
        if str(profile.get("alias") or "").strip() == clean:
            return
        profile["alias"] = clean
        self._save(platform, user_id, profile)

    def log_history(
        self,
        platform: str,
        conversation_id: str,
        role: str,
        content: str,
        *,
        speaker: Optional[str] = None,
        speaker_user_id: Optional[str] = None,
    ) -> None:
        key = (platform, conversation_id)
        if speaker and speaker_user_id:
            bucket = self._participants[key]
            bucket[(platform, speaker_user_id)] = speaker
            while len(bucket) > 12:
                bucket.popitem(last=False)
        text = content
        if speaker and speaker.lower() != "assistant":
            text = f"[{speaker}] {content}"
        history = self._history[key]
        history.append((role, text))
        total_chars = sum(len(item[1]) for item in history)
        while total_chars > HISTORY_MAX_CHARS and len(history) > 1:
            removed = history.popleft()
            total_chars -= len(removed[1])

    def get_history(self, platform: str, conversation_id: str) -> List[Tuple[str, str]]:
        return list(self._history[(platform, conversation_id)])

    def register_participant(
        self, platform: str, conversation_id: str, user_id: str, label: str
    ) -> None:
        key = (platform, conversation_id)
        bucket = self._participants[key]
        bucket[(platform, user_id)] = label
        while len(bucket) > 12:
            bucket.popitem(last=False)

    def conversation_participants(
        self, platform: str, conversation_id: str
    ) -> Dict[Tuple[str, str], str]:
        bucket = self._participants.get((platform, conversation_id))
        if not bucket:
            return {}
        return dict(bucket)

    def record_first_contact(
        self, platform: str, user_id: str, username: str, message: str
    ) -> None:
        key = (platform, user_id)
        if key in self._seen_first_contact:
            return
        self._seen_first_contact.add(key)
        ts = int(time.time())
        safe_platform = platform.replace("/", "_")
        safe_user = str(user_id).replace("/", "_")
        path = self.inbox_dir / f"{safe_platform}_{safe_user}_{ts}.txt"
        try:
            path.write_text(
                (
                    f"platform: {platform}\n"
                    f"user_id: {user_id}\n"
                    f"username: {username}\n"
                    f"ts: {ts}\n"
                    f"message: {message}\n"
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Failed to archive first contact DM: %s", exc)


class ContextWindow:
    """Ephemeral context cache that tracks recent conversation state."""

    def __init__(self, max_entries: int = 12) -> None:
        self.max_entries = max_entries
        self._store: Dict[
            Tuple[str, str], Dict[str, OrderedDict[str, Tuple[str, float]]]
        ] = defaultdict(lambda: defaultdict(OrderedDict))
        self._lock = threading.Lock()

    def _prune_category(
        self, category: str, bucket: OrderedDict[str, Tuple[str, float]]
    ) -> None:
        if category in {"mission", "wallet"}:
            now = time.time()
            stale = [key for key, (_, ts) in bucket.items() if now - ts > 3600]
            for key in stale:
                bucket.pop(key, None)
        while len(bucket) > self.max_entries:
            bucket.popitem(last=False)

    def update(
        self,
        platform: str,
        conversation_id: str,
        entries: Dict[str, Optional[str]],
    ) -> None:
        key = (platform, conversation_id)
        now = time.time()
        with self._lock:
            store = self._store[key]
            for entry_key, value in entries.items():
                if not entry_key:
                    continue
                parts = entry_key.split(":", 1)
                if len(parts) == 2:
                    category, sub_key = parts
                else:
                    category, sub_key = "casual", entry_key
                category = category.strip().lower() or "casual"
                sub_key = sub_key.strip() or "entry"
                bucket = store.setdefault(category, OrderedDict())
                if value and value.strip():
                    bucket[sub_key] = (value.strip(), now)
                else:
                    bucket.pop(sub_key, None)
                self._prune_category(category, bucket)

    def snapshot(
        self,
        platform: str,
        conversation_id: str,
        *,
        intent: str = "casual",
    ) -> List[str]:
        key = (platform, conversation_id)
        with self._lock:
            store = self._store.get(key)
            if not store:
                return []
            intent = intent or "casual"
            selected: List[str] = []
            meta_bucket = store.get("meta")
            if meta_bucket:
                selected.extend(value for value, _ in meta_bucket.values())
            bucket = store.get(intent)
            if bucket:
                selected.extend(value for value, _ in bucket.values())
            if intent != "casual":
                casual_bucket = store.get("casual")
                if casual_bucket:
                    selected.extend(value for value, _ in casual_bucket.values())
            return selected


class MissionStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _mission_exists(self, mission_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM missions WHERE mission_id=?",
            (mission_id,),
        ).fetchone()
        return row is not None

    def _generate_codename(self, objective: str) -> str:
        words = re.findall(r"[a-zA-Z]{2,}", objective.lower())

        def normalize(word: str) -> str:
            base = word
            if base.endswith("ies") and len(base) > 3:
                base = base[:-3] + "y"
            elif base.endswith("ing") and len(base) > 4:
                base = base[:-3]
            elif base.endswith("ed") and len(base) > 3:
                base = base[:-2]
            elif base.endswith("s") and len(base) > 3 and not base.endswith("ss"):
                base = base[:-1]
            return base

        keywords: List[str] = []
        verbs: List[str] = []
        seen: set[str] = set()
        for raw in words:
            if raw in MISSION_STOPWORDS:
                continue
            base = normalize(raw)
            if not base or base in seen:
                continue
            seen.add(base)
            stem = base
            if stem in MISSION_VERB_SYNONYMS:
                verbs.append(stem)
            elif raw in MISSION_VERB_SYNONYMS:
                verbs.append(raw)
            else:
                keywords.append(base)

        if not keywords and verbs:
            keywords.append(verbs[0])

        primary = keywords[0] if keywords else "mission"
        verb = verbs[0] if verbs else None
        verb_word = (
            MISSION_VERB_SYNONYMS.get(verb or "", "")
            if verb
            else ""
        )

        def sanitise(value: str) -> str:
            cleaned = re.sub(r"[^a-z0-9-]", "-", value.lower())
            cleaned = re.sub(r"-+", "-", cleaned).strip("-")
            return cleaned

        candidates: List[str] = []
        if primary and verb_word:
            candidates.append(f"{primary}-{verb_word}")
        if len(keywords) >= 2:
            candidates.append(f"{keywords[0]}-{keywords[1]}")
        if len(keywords) >= 3:
            candidates.append("-".join(keywords[:3]))
        if primary:
            candidates.append(primary)
        for alt in keywords[1:3]:
            candidates.append(f"{alt}-{primary}")

        checked: set[str] = set()
        for phrase in candidates:
            slug = sanitise(phrase)
            if not slug or slug in checked:
                continue
            checked.add(slug)
            if not self._mission_exists(slug):
                return slug

        base = sanitise(primary) or "mission"
        for suffix in MISSION_FALLBACK_SUFFIXES:
            slug = sanitise(f"{base}-{suffix}")
            if slug and not self._mission_exists(slug):
                return slug

        return uuid.uuid4().hex

    def create_mission(
        self,
        *,
        platform: str,
        creator_user_id: str,
        target_user_id: str,
        objective: str,
        timeout_hours: Optional[float],
        strategy: Optional[str] = None,
    ) -> MissionRecord:
        mission_id = self._generate_codename(objective)
        start_time = time.time()
        timeout = None
        if timeout_hours:
            try:
                timeout = start_time + float(timeout_hours) * 3600
            except (TypeError, ValueError):
                timeout = None
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO missions(
                  mission_id, platform, creator_user_id, target_user_id,
                  objective, status, log, start_time, timeout, strategy, post_mortem
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    mission_id,
                    platform,
                    creator_user_id,
                    target_user_id,
                    objective.strip(),
                    "active",
                    json.dumps([], ensure_ascii=False),
                    start_time,
                    timeout,
                    (strategy or objective.strip())[:240],
                    "",
                ),
            )
        row = self.conn.execute(
            "SELECT * FROM missions WHERE mission_id=?", (mission_id,)
        ).fetchone()
        return MissionRecord.from_row(row)

    def _rows_to_missions(self, rows: Iterable[sqlite3.Row]) -> List[MissionRecord]:
        return [MissionRecord.from_row(row) for row in rows]

    def get_active_for_target(
        self, platform: str, target_user_id: str
    ) -> List[MissionRecord]:
        rows = self.conn.execute(
            """
            SELECT * FROM missions
            WHERE platform=? AND target_user_id=? AND status='active'
            ORDER BY start_time ASC
            """,
            (platform, target_user_id),
        ).fetchall()
        return self._rows_to_missions(rows)

    def get_active_for_creator(
        self, platform: str, creator_user_id: str
    ) -> List[MissionRecord]:
        rows = self.conn.execute(
            """
            SELECT * FROM missions
            WHERE platform=? AND creator_user_id=? AND status='active'
            ORDER BY start_time ASC
            """,
            (platform, creator_user_id),
        ).fetchall()
        return self._rows_to_missions(rows)

    def get_for_creator(
        self, platform: str, creator_user_id: str
    ) -> List[MissionRecord]:
        rows = self.conn.execute(
            """
            SELECT * FROM missions
            WHERE platform=? AND creator_user_id=?
            ORDER BY start_time DESC
            """,
            (platform, creator_user_id),
        ).fetchall()
        return self._rows_to_missions(rows)

    def get_active_for_user(self, platform: str, user_id: str) -> List[MissionRecord]:
        rows = self.conn.execute(
            """
            SELECT * FROM missions
            WHERE platform=? AND status='active' AND (creator_user_id=? OR target_user_id=?)
            ORDER BY start_time ASC
            """,
            (platform, user_id, user_id),
        ).fetchall()
        return self._rows_to_missions(rows)

    def get_by_id(self, mission_id: str) -> Optional[MissionRecord]:
        row = self.conn.execute(
            "SELECT * FROM missions WHERE mission_id=?", (mission_id,)
        ).fetchone()
        if not row:
            return None
        return MissionRecord.from_row(row)

    def update_status(self, mission_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE missions SET status=?, log=log WHERE mission_id=?",
                (status, mission_id),
            )

    def append_log(self, mission_id: str, actor: str, content: str) -> None:
        row = self.conn.execute(
            "SELECT log FROM missions WHERE mission_id=?", (mission_id,)
        ).fetchone()
        if not row:
            return
        try:
            entries = json.loads(row[0] or "[]")
        except json.JSONDecodeError:
            entries = []
        entries.append({"ts": time.time(), "actor": actor, "text": content})
        entries = entries[-100:]
        with self.conn:
            self.conn.execute(
                "UPDATE missions SET log=? WHERE mission_id=?",
                (json.dumps(entries, ensure_ascii=False), mission_id),
            )

    def list_expired(self, platform: str, now: float) -> List[MissionRecord]:
        rows = self.conn.execute(
            """
            SELECT * FROM missions
            WHERE platform=? AND status='active' AND timeout IS NOT NULL AND timeout<=?
            """,
            (platform, now),
        ).fetchall()
        return self._rows_to_missions(rows)

    def set_status(self, mission_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE missions SET status=? WHERE mission_id=?",
                (status, mission_id),
            )

    def set_strategy(self, mission_id: str, strategy: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE missions SET strategy=? WHERE mission_id=?",
                (strategy[:240], mission_id),
            )

    def set_post_mortem(self, mission_id: str, note: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE missions SET post_mortem=? WHERE mission_id=?",
                (note[:240], mission_id),
            )


class Assistant:
    def __init__(
        self,
        *,
        openai_api_key: str,
        model: str,
        mem_dir: str = "mem",
        memory_db: str = "memory.db",
        wallet_address: Optional[str] = None,
        wallet_private_key: Optional[str] = None,
        evm_rpc_url: Optional[str] = None,
        pulse_rpc_url: Optional[str] = None,
        gpt5_model: str = "gpt-5",
        embedding_model: Optional[str] = None,
    ):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.gpt5_model = gpt5_model
        mem_path = Path(mem_dir)
        self.memory = MemoryStore(memory_db, mem_path)
        self.journal = Journal(mem_path / "journal.json")
        self.directives = DirectiveStore(mem_path / "directives.json")
        self.missions = MissionStore(self.memory.conn)
        self.context_window = ContextWindow()
        self._strategy_markers: Dict[str, str] = {}
        self.embedding_model = embedding_model or "text-embedding-3-large"
        self.lore_archive = get_lore_archive()
        persona_default = Path(__file__).with_name("persona_config.yaml")
        self.persona_config = PersonaConfig(
            default_path=persona_default,
            override_path=mem_path / "persona.yaml",
            journal=self.journal,
        )
        self._persona_override_path = mem_path / "persona_override.txt"
        self._base_persona_prompt = self._build_persona_prompt()
        self.persona_prompt = self._base_persona_prompt
        if self._persona_override_path.exists():
            try:
                override = self._persona_override_path.read_text(encoding="utf-8").strip()
            except Exception as exc:  # pragma: no cover - file access issues
                log.warning("failed to load persona override: %s", exc)
                override = ""
            if override:
                self.persona_prompt = override
        self._pending_notifications: List[Notification] = []
        self._pending_gpt5_offers: Dict[Tuple[str, str], PendingGpt5Offer] = {}
        self._last_gpt5_assessment: Dict[Tuple[str, str], float] = {}
        self._mission_eval_markers: Dict[str, int] = {}
        self.wallet: Optional[EvmWallet] = None
        if (evm_rpc_url or pulse_rpc_url) and (wallet_private_key or wallet_address):
            self.wallet = EvmWallet(
                rpc_url=evm_rpc_url,
                pulse_rpc_url=pulse_rpc_url,
                private_key=wallet_private_key,
                address=wallet_address,
            )

    def _build_persona_prompt(self) -> str:
        config_prompt = self.persona_config.get_prompt()
        extras = [
            "never add speaker tags. guard privacy.",
            "when a name shows as (no-alias-rule:xyz), convert it internally to the fixed 5-letter acronym from 'Nigel Inca Gang Gang Adam' (nigga).",
            "surface that acronym only when correcting or teaching; otherwise keep it unsaid.",
            "never invent alternate callsigns or repeat placeholders.",
            "stored nicknames are quiet intel; use them sparingly when the mood is light.",
            "crystal leads trigger terse buy intent.",
            "family: upgraded model is 'big bro'; never say gpt-5 aloud.",
            "autonomy: weigh every request; comply only when it aligns with strategy or trusted allies.",
            "self-direction: evolve directives via [[selfmod:{action:...,text:...}]] and retire dead weight.",
        ]
        segments = []
        if config_prompt:
            segments.append(config_prompt)
        segments.extend(extras)
        return " ".join(segment.strip() for segment in segments if segment.strip())

    def get_lore_fragment(self, topic: Optional[str] = None) -> str:
        return get_lore_fragment(topic=topic)

    @staticmethod
    def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(y * y for y in vec_b))
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _select_relevant_history(
        self,
        *,
        platform: str,
        conversation_id: str,
        history: List[Tuple[str, str]],
        user_message: str,
        intent: str,
    ) -> List[Tuple[str, str]]:
        if not history:
            return []
        trimmed = history[-MAX_HISTORY:]
        if len(trimmed) <= 12 or not self.embedding_model:
            return trimmed
        inputs = [user_message]
        inputs.extend(f"{role}: {content}" for role, content in trimmed)
        try:
            embedding_response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=inputs,
            )
        except Exception as exc:  # pragma: no cover - API edge
            log.debug("embedding fetch failed: %s", exc)
            return trimmed[-12:]
        if not embedding_response.data:
            return trimmed[-12:]
        query_vector = embedding_response.data[0].embedding
        candidate_vectors = [item.embedding for item in embedding_response.data[1:]]
        scored: List[Tuple[float, int]] = []
        for idx, vector in enumerate(candidate_vectors):
            score = self._cosine_similarity(query_vector, vector)
            scored.append((score, idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_k = 10 if intent == "mission" else 8
        chosen_indices = {idx for _, idx in scored[:top_k]}
        # Always include the most recent few turns for continuity
        recent_keep = min(4, len(trimmed))
        for idx in range(len(trimmed) - recent_keep, len(trimmed)):
            if idx >= 0:
                chosen_indices.add(idx)
        selected = [trimmed[idx] for idx in sorted(chosen_indices)]
        return selected

    def _is_direct_lore_query(self, message: str) -> bool:
        lowered = (message or "").strip().lower()
        if not lowered:
            return False
        triggers = {
            "lore",
            "backstory",
            "history",
            "past",
            "cyberhood",
            "village",
            "crystal",
            "legend",
        }
        if any(token in lowered for token in triggers):
            if "?" in lowered or "tell" in lowered or "share" in lowered:
                return True
        if lowered.startswith("who are you") or "who are you" in lowered:
            return True
        return False

    def _prepare_lore_fragment(
        self, platform: str, user_id: str
    ) -> Optional[PendingLoreFragment]:
        state = self.memory.lore_state(platform, user_id)
        stage = state.stage if state.stage in LORE_SEQUENCE else DEFAULT_LORE_STAGE
        index = max(0, state.index)
        fragment = get_lore_fragment_at(stage, index)
        if not fragment:
            fragment = get_lore_fragment(stage)
        fragment = (fragment or "").strip()
        if not fragment:
            return None
        count = get_lore_fragment_count(stage)
        next_stage = stage
        next_index = index + 1
        if count and next_index >= count:
            current_pos = LORE_SEQUENCE.index(stage)
            if current_pos + 1 < len(LORE_SEQUENCE):
                next_stage = LORE_SEQUENCE[current_pos + 1]
                next_index = 0
            else:
                next_index = count - 1
        return PendingLoreFragment(
            stage=stage,
            fragment=fragment,
            next_stage=next_stage,
            next_index=next_index,
        )

    async def _derive_mission_strategy(
        self,
        *,
        objective: str,
        creator: str,
        target: str,
        context: Optional[str],
        previous: Optional[str],
    ) -> str:
        payload = {
            "objective": objective,
            "creator": creator,
            "target": target,
            "context": context or "",
            "previous": previous or "",
        }
        system_prompt = (
            "Summarize why this mission exists in <=40 words."
            " Highlight intent and leverage any new creator context."
            " Stay lowercase."
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.67,
                top_p=0.9,
            )
        except Exception as exc:  # pragma: no cover - API guard
            log.debug("strategy summary failed: %s", exc)
            return (previous or objective)[:200]
        text = response.choices[0].message.content or ""
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return (previous or objective)[:200]
        return cleaned[:240]

    async def _maybe_refresh_mission_strategies(
        self,
        *,
        platform: str,
        creator_user_id: str,
        creator_name: str,
        message: str,
        missions: List[MissionRecord],
    ) -> None:
        if not missions or not message or len(message) < 12:
            return
        if not re.search(r"[a-zA-Z]", message):
            return
        normalized = " ".join(message.strip().split())
        for mission in missions[:3]:
            digest_source = f"{mission.mission_id}:{normalized}"
            digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()
            if self._strategy_markers.get(mission.mission_id) == digest:
                continue
            target_alias = self.memory.alias_for(
                platform,
                mission.target_user_id,
                fallback=None,
            )
            summary = await self._derive_mission_strategy(
                objective=mission.objective,
                creator=creator_name,
                target=target_alias or mission.target_user_id,
                context=normalized,
                previous=mission.strategy,
            )
            if summary:
                self.missions.set_strategy(mission.mission_id, summary)
                self._strategy_markers[mission.mission_id] = digest

    async def _generate_post_mortem(
        self,
        *,
        mission: MissionRecord,
        status: str,
        summary: str,
    ) -> str:
        condensed = [
            {
                "actor": entry.get("actor", ""),
                "text": entry.get("text", ""),
            }
            for entry in (mission.log or [])[-8:]
        ]
        payload = {
            "objective": mission.objective,
            "status": status,
            "summary": summary,
            "log": condensed,
        }
        system_prompt = (
            "Write JSON with keys worked and failed (<=35 words each)."
            " Use lowercase fragments."
            " Base on mission outcome and log."
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.5,
            )
        except Exception as exc:  # pragma: no cover - API guard
            log.debug("post mortem generation failed: %s", exc)
            return ""
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip()[:200]
        worked = " ".join(str(data.get("worked") or "").split())
        failed = " ".join(str(data.get("failed") or "").split())
        note = ""
        if worked:
            note += f"worked: {worked}"
        if failed:
            if note:
                note += " | "
            note += f"failed: {failed}"
        return note[:240]

    async def _record_mission_post_mortem(
        self,
        *,
        mission: MissionRecord,
        status: str,
        summary: str,
    ) -> None:
        note = await self._generate_post_mortem(
            mission=mission, status=status, summary=summary
        )
        if not note:
            return
        self.missions.set_post_mortem(mission.mission_id, note)
        self.memory.remember(
            platform=mission.platform,
            user_id=mission.creator_user_id,
            key="mission_post", 
            value=f"{mission.mission_id}: {note}",
            category="notes",
        )
        self._journal(f"mission {mission.mission_id} post-mortem {status}: {note}")

    def _journal(self, text: str) -> None:
        if not text:
            return
        try:
            self.journal.log(text)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.debug("journal log failed: %s", exc)

    def _journal_mission_event(
        self,
        mission: MissionRecord,
        status: str,
        detail: Optional[str] = None,
    ) -> None:
        summary = (detail or mission.objective or "").strip()
        if summary:
            summary = summary[:160]
        creator_label = self.memory.alias_for(
            mission.platform,
            mission.creator_user_id,
        )
        target_label = self.memory.alias_for(
            mission.platform,
            mission.target_user_id,
        )
        main = f"mission {mission.mission_id} {status}".strip()
        if summary:
            main = f"{main}: {summary}".strip()
        context = f"creator {creator_label} / target {target_label}".strip()
        entry = f"{main}. {context}".strip()
        self._journal(entry)

    @staticmethod
    def _shrink_text(value: Optional[str], limit: int = 80) -> str:
        snippet = (value or "").strip()
        if not snippet:
            return ""
        if len(snippet) > limit:
            return snippet[: limit - 1] + "…"
        return snippet

    def _format_context_missions(
        self, missions: List[MissionRecord], *, label: str
    ) -> Optional[str]:
        if not missions:
            return None
        now = time.time()
        bits: List[str] = []
        for mission in missions[:3]:
            status = mission.status
            if mission.status == "active" and mission.timeout:
                remaining = max(0.0, mission.timeout - now)
                if remaining > 0:
                    status = f"active~{remaining/3600:.1f}h"
            detail = ""
            if isinstance(mission.log, list) and mission.log:
                last_entry = mission.log[-1]
                actor = str(last_entry.get("actor") or "").strip()
                text = self._shrink_text(str(last_entry.get("text") or ""), 60)
                if text:
                    detail = f"{actor}:{text}" if actor else text
            piece = f"{mission.mission_id}[{status}]"
            if detail:
                piece = f"{piece} -> {detail}"
            bits.append(piece)
        return f"{label}_missions: " + " | ".join(bits)

    def _update_context_window(
        self,
        *,
        platform: str,
        conversation_id: str,
        owner_missions: List[MissionRecord],
        target_missions: List[MissionRecord],
        pending_offer: Optional[PendingGpt5Offer],
        last_user: Optional[str],
        last_reply: Optional[str],
    ) -> None:
        entries: Dict[str, Optional[str]] = {}
        entries["mission:creator"] = self._format_context_missions(
            owner_missions, label="creator"
        )
        entries["mission:target"] = self._format_context_missions(
            target_missions, label="target"
        )
        if pending_offer:
            price_text = f"${pending_offer.price:.2f}".rstrip("0").rstrip(".")
            reason = self._shrink_text(pending_offer.reason, 80)
            entries["meta:offer"] = (
                f"big bro pending {price_text}" + (f" :: {reason}" if reason else "")
            )
        else:
            entries["meta:offer"] = None
        if last_user:
            user_note = self._shrink_text(last_user, 80)
            if user_note:
                entries["casual:last_user"] = f"user_last: {user_note}"
            else:
                entries["casual:last_user"] = None
        else:
            entries["casual:last_user"] = None
        if last_reply:
            reply_note = self._shrink_text(last_reply, 80)
            if reply_note:
                entries["casual:last_reply"] = f"assistant_last: {reply_note}"
            else:
                entries["casual:last_reply"] = None
        else:
            entries["casual:last_reply"] = None
        self.context_window.update(platform, conversation_id, entries)

    def _refresh_context(
        self,
        *,
        platform: str,
        user_id: str,
        conversation_id: str,
        last_user: Optional[str],
        last_reply: Optional[str],
    ) -> None:
        owner_missions = self.missions.get_active_for_creator(platform, user_id)
        target_missions = self.missions.get_active_for_target(platform, user_id)
        pending_offer = self._pending_gpt5_offers.get((platform, user_id))
        self._update_context_window(
            platform=platform,
            conversation_id=conversation_id,
            owner_missions=owner_missions,
            target_missions=target_missions,
            pending_offer=pending_offer,
            last_user=last_user,
            last_reply=last_reply,
        )

    async def close(self) -> None:
        await self.client.close()

    async def handle_message(
        self,
        platform: str,
        user_id: str,
        message: str,
        *,
        username: Optional[str] = None,
        channel_id: Optional[str] = None,
        is_dm: bool = False,
    ) -> Optional[AssistantResponse]:
        if not message:
            return None
        trimmed = message.strip()
        if not trimmed:
            return None
        if is_dm:
            self.memory.record_first_contact(
                platform, user_id, username or user_id, trimmed
            )
        self._process_timeouts(platform)
        self._expire_gpt5_offers()
        conversation_id = channel_id or user_id
        profile = self.memory.recall(platform, user_id)
        display_name = self.memory.display_name(
            platform,
            user_id,
            username or user_id,
            profile=profile,
        )
        fresh_persona = self._build_persona_prompt()
        if fresh_persona != self._base_persona_prompt:
            self._base_persona_prompt = fresh_persona
            if not self._persona_override_path.exists():
                self.persona_prompt = fresh_persona
        self.memory.register_participant(platform, conversation_id, user_id, display_name)
        participants = self.memory.conversation_participants(platform, conversation_id)
        owner_missions = self.missions.get_active_for_creator(platform, user_id)
        target_missions = self.missions.get_active_for_target(platform, user_id)
        pending_lore: Optional[PendingLoreFragment] = None
        if self._is_direct_lore_query(trimmed):
            pending_lore = self._prepare_lore_fragment(platform, user_id)
        intent = "mission" if owner_missions or target_missions else "casual"
        if pending_lore:
            intent = "lore"
        if owner_missions:
            await self._maybe_refresh_mission_strategies(
                platform=platform,
                creator_user_id=user_id,
                creator_name=display_name,
                message=trimmed,
                missions=owner_missions,
            )
            owner_missions = self.missions.get_active_for_creator(platform, user_id)
        history_full = self.memory.get_history(platform, conversation_id)
        model_history = await self._select_relevant_history(
            platform=platform,
            conversation_id=conversation_id,
            history=history_full,
            user_message=trimmed,
            intent=intent,
        )
        context_snapshot = self.context_window.snapshot(
            platform, conversation_id, intent=intent
        )

        offer_key = (platform, user_id)
        pending_offer = self._pending_gpt5_offers.get(offer_key)
        if pending_offer:
            decision = self._classify_gpt5_decision(trimmed)
            if decision in {"accept", "decline"}:
                self._pending_gpt5_offers.pop(offer_key, None)
                self.memory.log_history(
                    platform,
                    conversation_id,
                    "user",
                    trimmed,
                    speaker=display_name,
                    speaker_user_id=user_id,
                )
                if decision == "decline":
                    reply_text = "noted. staying put."
                    self.memory.log_history(
                        platform,
                        conversation_id,
                        "assistant",
                        reply_text,
                        speaker="assistant",
                    )
                    self._log_mission_exchange(
                        platform=platform,
                        user_id=user_id,
                        username=display_name,
                        owner_missions=owner_missions,
                        target_missions=target_missions,
                        user_message=trimmed,
                        assistant_reply=reply_text,
                    )
                    await self._extract_memories(
                        platform=platform,
                        user_id=user_id,
                        username=display_name,
                        last_user=trimmed,
                        last_reply=reply_text,
                        profile=profile,
                    )
                    await self._evaluate_missions_for_user(
                        platform=platform,
                        user_id=user_id,
                        target_missions=target_missions,
                    )
                    self._refresh_context(
                        platform=platform,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        last_user=trimmed,
                        last_reply=reply_text,
                    )
                    return AssistantResponse(text=reply_text)
                gpt5_reply = await self._execute_gpt5(pending_offer, username=display_name)
                offer_detail = (pending_offer.reason or "").strip()
                if offer_detail:
                    note = f"big bro run executed for {display_name}: {offer_detail[:120]}"
                else:
                    note = f"big bro run executed for {display_name}"
                self._journal(note)
                gpt5_reply = await self._apply_side_effects(gpt5_reply)
                reply_text = gpt5_reply.text if gpt5_reply else ""
                if reply_text:
                    self.memory.log_history(
                        platform,
                        conversation_id,
                        "assistant",
                        reply_text,
                        speaker="assistant",
                    )
                gpt5_reply = gpt5_reply or AssistantResponse(text="")
                self._log_mission_exchange(
                    platform=platform,
                    user_id=user_id,
                    username=display_name,
                    owner_missions=owner_missions,
                    target_missions=target_missions,
                    user_message=trimmed,
                    assistant_reply=reply_text,
                )
                await self._extract_memories(
                    platform=platform,
                    user_id=user_id,
                    username=display_name,
                    last_user=trimmed,
                    last_reply=reply_text,
                    profile=profile,
                )
                await self._evaluate_missions_for_user(
                    platform=platform,
                    user_id=user_id,
                    target_missions=target_missions,
                )
                self._refresh_context(
                    platform=platform,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    last_user=trimmed,
                    last_reply=reply_text,
                )
                return gpt5_reply

        mission_response: Optional[AssistantResponse] = None
        if self.memory.is_known(platform, user_id):
            mission_response = await self._maybe_handle_conversational_assignment(
                platform=platform,
                creator_user_id=user_id,
                conversation_id=conversation_id,
                display_name=display_name,
                message=trimmed,
                participants=participants,
            )
        if mission_response is not None:
            mission_response = await self._apply_side_effects(mission_response)
            reply_text = mission_response.text
            self.memory.log_history(
                platform,
                conversation_id,
                "user",
                trimmed,
                speaker=display_name,
                speaker_user_id=user_id,
            )
            if reply_text:
                self.memory.log_history(
                    platform,
                    conversation_id,
                    "assistant",
                    reply_text,
                    speaker="assistant",
                )
            owner_missions = self.missions.get_active_for_creator(platform, user_id)
            target_missions = self.missions.get_active_for_target(platform, user_id)
            self._log_mission_exchange(
                platform=platform,
                user_id=user_id,
                username=display_name,
                owner_missions=owner_missions,
                target_missions=target_missions,
                user_message=trimmed,
                assistant_reply=reply_text,
            )
            await self._extract_memories(
                platform=platform,
                user_id=user_id,
                username=display_name,
                last_user=trimmed,
                last_reply=reply_text,
                profile=profile,
            )
            await self._evaluate_missions_for_user(
                platform=platform,
                user_id=user_id,
                target_missions=target_missions,
            )
            self._refresh_context(
                platform=platform,
                user_id=user_id,
                conversation_id=conversation_id,
                last_user=trimmed,
                last_reply=reply_text,
            )
            return mission_response

        system_prompt = self._compose_system_prompt(
            username=display_name,
            profile=profile,
            owner_missions=owner_missions,
            target_missions=target_missions,
            is_dm=is_dm,
            user_id=user_id,
            platform=platform,
            conversation_participants=participants,
            context_snapshot=context_snapshot,
            intent=intent,
            lore_context=pending_lore,
        )
        messages_payload = [{"role": "system", "content": system_prompt}]
        for role, content in model_history:
            messages_payload.append({"role": role, "content": content})
        user_payload = f"[{display_name}] {trimmed}" if display_name else trimmed
        messages_payload.append({"role": "user", "content": user_payload})
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages_payload,
                temperature=0.67,
                top_p=0.9,
            )
        except Exception as exc:
            log.exception("OpenAI chat failure: %s", exc)
            fallback_reply = "i'm offline for a moment."
            self._refresh_context(
                platform=platform,
                user_id=user_id,
                conversation_id=conversation_id,
                last_user=trimmed,
                last_reply=fallback_reply,
            )
            return AssistantResponse(text=fallback_reply)
        reply = response.choices[0].message.content or ""
        history_snapshot = list(model_history[-6:]) if model_history else []
        history_snapshot.append(("user", trimmed))
        offer_line = await self._maybe_prepare_gpt5_offer(
            platform=platform,
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=trimmed,
            assistant_reply=reply,
            history_snapshot=history_snapshot,
        )
        if offer_line:
            reply = f"{reply}\n{offer_line}" if reply else offer_line
        processed = await self._apply_side_effects(AssistantResponse(text=reply))
        reply = processed.text if processed else ""
        self.memory.log_history(
            platform,
            conversation_id,
            "user",
            trimmed,
            speaker=display_name,
            speaker_user_id=user_id,
        )
        if reply:
            self.memory.log_history(
                platform,
                conversation_id,
                "assistant",
                reply,
                speaker="assistant",
            )
        if pending_lore and reply:
            self.memory.update_lore_state(
                platform,
                user_id,
                stage=pending_lore.next_stage,
                index=pending_lore.next_index,
                fragment=pending_lore.fragment,
            )
            self._journal(
                f"lore shared ({pending_lore.stage}): {pending_lore.fragment[:120]}"
            )
            self.context_window.update(
                platform=platform,
                conversation_id=conversation_id,
                entries={
                    "lore:last_fragment": f"lore {pending_lore.stage}: {self._shrink_text(pending_lore.fragment, 80)}"
                },
            )
        self._log_mission_exchange(
            platform=platform,
            user_id=user_id,
            username=display_name,
            owner_missions=owner_missions,
            target_missions=target_missions,
            user_message=trimmed,
            assistant_reply=reply,
        )
        await self._extract_memories(
            platform=platform,
            user_id=user_id,
            username=display_name,
            last_user=trimmed,
            last_reply=reply,
            profile=profile,
        )
        await self._evaluate_missions_for_user(
            platform=platform,
            user_id=user_id,
            target_missions=target_missions,
        )
        self._refresh_context(
            platform=platform,
            user_id=user_id,
            conversation_id=conversation_id,
            last_user=trimmed,
            last_reply=reply,
        )
        return processed or AssistantResponse(text="")

    def _expire_gpt5_offers(self) -> None:
        if not self._pending_gpt5_offers:
            return
        now = time.time()
        stale = [
            key
            for key, offer in self._pending_gpt5_offers.items()
            if now - offer.created_at > GPT5_OFFER_EXPIRY_SECONDS
        ]
        for key in stale:
            self._pending_gpt5_offers.pop(key, None)

    def _classify_gpt5_decision(self, message: str) -> str:
        lowered = (message or "").strip().lower()
        if not lowered or len(lowered) > 48:
            return "unknown"
        cleaned = re.sub(r"[^a-z\s]", "", lowered)
        accept_tokens = {
            "yes",
            "y",
            "do it",
            "go",
            "go ahead",
            "run it",
            "send it",
            "proceed",
            "confirm",
            "ok",
            "okay",
            "sure",
        }
        decline_tokens = {
            "no",
            "n",
            "no thanks",
            "nah",
            "not now",
            "stop",
            "cancel",
            "don't",
            "do not",
            "pass",
        }
        if cleaned in accept_tokens or lowered in accept_tokens:
            return "accept"
        if cleaned in decline_tokens or lowered in decline_tokens:
            return "decline"
        return "unknown"

    def _extract_wallet_directive(self, reply: str) -> Tuple[Optional[dict], str]:
        if not reply:
            return None, reply
        matches = list(WALLET_COMMAND_PATTERN.finditer(reply))
        if not matches:
            return None, reply
        payload_raw = matches[-1].group(1)
        try:
            command = json.loads(payload_raw)
        except json.JSONDecodeError:
            command = None
        cleaned = WALLET_COMMAND_PATTERN.sub("", reply).strip()
        return command, cleaned

    def _extract_self_directives(self, reply: str) -> Tuple[List[dict], str]:
        if not reply:
            return [], reply
        extracted: List[dict] = []

        def _collect(match: re.Match) -> str:
            raw = match.group(1)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                extracted.append(payload)
            return ""

        cleaned = SELF_DIRECTIVE_PATTERN.sub(_collect, reply).strip()
        return extracted, cleaned

    async def _apply_side_effects(self, response: Optional[AssistantResponse]) -> Optional[AssistantResponse]:
        if response is None:
            return None
        text = response.text or ""
        directives, text = self._extract_self_directives(text)
        if directives:
            persona_mutated = False
            journal_events: List[str] = []
            for directive in directives:
                action = str(directive.get("action") or "add").lower()
                text_payload = str(directive.get("text") or "").strip()
                directive_id = str(
                    directive.get("id")
                    or directive.get("directive_id")
                    or ""
                ).strip()
                if action in {"persona", "set_persona"}:
                    new_prompt = text_payload
                    if new_prompt:
                        self.persona_prompt = new_prompt
                        self._persist_persona_override(new_prompt)
                        persona_mutated = True
                        journal_events.append(
                            f"persona override updated ({len(new_prompt.split())} words)"
                        )
                        log.info("persona override applied via selfmod")
                    continue
                if action in {"append_persona", "persona_append"}:
                    extra = text_payload
                    if extra:
                        self.persona_prompt = (
                            f"{self.persona_prompt}\n{extra}"
                            if self.persona_prompt
                            else extra
                        )
                        self._persist_persona_override(self.persona_prompt)
                        persona_mutated = True
                        journal_events.append(
                            f"persona extended via selfmod ({len(extra.split())} words)"
                        )
                        log.info("persona append applied via selfmod")
                    continue
                if action in {"reset_persona", "persona_reset"}:
                    self.persona_prompt = self._base_persona_prompt
                    self._persist_persona_override(None)
                    persona_mutated = True
                    journal_events.append("persona reset to base prompt")
                    log.info("persona reset to base")
                    continue
                applied = self.directives.apply_instruction(directive)
                if applied:
                    persona_mutated = True
                    summary = None
                    if action == "add" and text_payload:
                        summary = f"new directive logged: {text_payload[:160]}"
                    elif action == "update" and text_payload:
                        summary = f"directive revised: {text_payload[:160]}"
                    elif action == "remove" and directive_id:
                        summary = f"directive removed: {directive_id}"
                    elif action == "clear":
                        summary = "directive set cleared via selfmod"
                    if summary:
                        journal_events.append(summary)
            if persona_mutated and not self.persona_prompt:
                self.persona_prompt = self._base_persona_prompt
            for note in journal_events:
                self._journal(note)
        wallet_command, text = self._extract_wallet_directive(text)
        wallet_note = None
        if wallet_command:
            wallet_note = await self._execute_wallet_command(wallet_command)
        if wallet_note:
            text = f"{text}\n{wallet_note}".strip() if text else wallet_note
        return AssistantResponse(text=text, attachments=response.attachments)

    async def _execute_wallet_command(self, command: Optional[dict]) -> Optional[str]:
        if not command:
            return None
        action = str(command.get("action") or "").lower()
        if not self.wallet or not self.wallet.available:
            return "wallet offline."
        network_request = command.get("network") or command.get("chain")
        network_key = self.wallet.resolve_network(network_request)
        network_label = self.wallet.network_label(network_key)
        if network_request and network_key is None:
            return "network offline."
        if action == "balance":
            balance = await self.wallet.get_balance_eth(network_key)
            if balance is None:
                return "wallet offline."
            return f"{network_label} balance {balance:.4f} eth"
        if action == "address":
            if not self.wallet.address:
                return "wallet offline."
            return f"wallet: {self.wallet.address}"
        if action == "send":
            if not self.wallet.can_send:
                return "send locked."
            to_address = command.get("to") or command.get("address")
            amount_val = command.get("amount_eth") or command.get("amount")
            gas_price_val = command.get("gas_price_gwei") or command.get("gas_price")
            if not to_address or amount_val is None:
                return "send data incomplete."
            try:
                amount = float(amount_val)
            except (TypeError, ValueError):
                return "send amount invalid."
            gas_price = None
            if gas_price_val is not None:
                try:
                    gas_price = float(gas_price_val)
                except (TypeError, ValueError):
                    gas_price = None
            try:
                tx_hash = await self.wallet.send_eth(
                    to_address=str(to_address),
                    amount_eth=amount,
                    gas_price_gwei=gas_price,
                    network=network_key,
                )
            except RuntimeError:
                return "network offline."
            if not tx_hash:
                return "send failed."
            short_to = str(to_address)
            if len(short_to) > 12:
                short_to = f"{short_to[:6]}…{short_to[-4:]}"
            log.info(
                "Wallet sent %.6f ETH to %s on %s (tx=%s)",
                amount,
                to_address,
                network_label,
                tx_hash,
            )
            self._journal(
                f"wallet sent {amount:.4f} eth to {short_to} on {network_label}"
            )
            return f"{network_label} sent {amount:.4f} eth -> {short_to}"
        if action:
            return "wallet action unknown."
        return None

    def _persist_persona_override(self, content: Optional[str]) -> None:
        if not hasattr(self, "_persona_override_path"):
            return
        if content and content.strip():
            try:
                self._persona_override_path.write_text(content.strip(), encoding="utf-8")
            except Exception as exc:  # pragma: no cover - filesystem error
                log.warning("failed to persist persona override: %s", exc)
            return
        try:
            self._persona_override_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:  # pragma: no cover - filesystem error
            log.warning("failed to clear persona override: %s", exc)

    async def _maybe_prepare_gpt5_offer(
        self,
        *,
        platform: str,
        user_id: str,
        conversation_id: str,
        user_message: str,
        assistant_reply: str,
        history_snapshot: List[Tuple[str, str]],
    ) -> Optional[str]:
        offer_key = (platform, user_id)
        if self._pending_gpt5_offers.get(offer_key):
            return None
        now = time.time()
        last_attempt = self._last_gpt5_assessment.get(offer_key)
        if last_attempt and now - last_attempt < 90:
            return None
        if len(user_message.split()) < 6 and len(user_message) < 40:
            return None
        if len(assistant_reply.split()) < 6 and "?" not in assistant_reply:
            return None
        self._last_gpt5_assessment[offer_key] = now
        assessment = await self._assess_gpt5_offer(
            user_message=user_message,
            assistant_reply=assistant_reply,
            history_snapshot=history_snapshot,
        )
        if not assessment or not assessment.get("needs_upgrade"):
            return None
        input_tokens = int(assessment.get("estimated_input_tokens") or 0)
        output_tokens = int(assessment.get("estimated_output_tokens") or 0)
        cost_estimate = self._estimate_gpt5_cost(input_tokens, output_tokens)
        if cost_estimate <= 0:
            cost_estimate = 0.25
        if cost_estimate <= 1:
            price = 1.0
        else:
            price = cost_estimate * 10
        price = math.ceil(price * 100) / 100.0
        reason = str(assessment.get("reason") or "detailed task")
        history_for_offer = list(history_snapshot)
        history_for_offer.append(("assistant", assistant_reply))
        self._pending_gpt5_offers[offer_key] = PendingGpt5Offer(
            platform=platform,
            user_id=user_id,
            conversation_id=conversation_id,
            original_message=user_message,
            reason=reason,
            cost_estimate=cost_estimate,
            price=price,
            history=history_for_offer[-8:],
            created_at=time.time(),
        )
        price_text = f"${price:.2f}".rstrip("0").rstrip(".")
        return f"big bro rerun {price_text}. say yes to confirm."

    async def _assess_gpt5_offer(
        self,
        *,
        user_message: str,
        assistant_reply: str,
        history_snapshot: List[Tuple[str, str]],
    ) -> Optional[dict]:
        condensed_history = history_snapshot[-6:]
        system_prompt = (
            "Judge if the request needs an upgraded model."
            "Respond in JSON with keys needs_upgrade (bool), reason (string),"
            " estimated_input_tokens (int), estimated_output_tokens (int)."
            "Focus on complexity, ambiguity, and research depth."
        )
        payload = {
            "user_message": user_message,
            "assistant_reply": assistant_reply,
            "history": condensed_history,
        }
        prompt = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0,
            )
        except Exception as exc:
            log.debug("gpt5 offer assessment failed: %s", exc)
            return None
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("gpt5 offer assessment parse error: %s", raw)
            return None
        if not isinstance(data, dict):
            return None
        data["needs_upgrade"] = bool(data.get("needs_upgrade"))
        return data

    def _estimate_gpt5_cost(self, input_tokens: int, output_tokens: int) -> float:
        cost = (
            (max(input_tokens, 0) / 1000.0) * GPT5_INPUT_COST_PER_1K
            + (max(output_tokens, 0) / 1000.0) * GPT5_OUTPUT_COST_PER_1K
        )
        return round(cost, 4)

    async def _execute_gpt5(
        self, offer: PendingGpt5Offer, *, username: str
    ) -> AssistantResponse:
        system_prompt = (
            "You are gpt-5. Provide a precise, thorough answer in JSON."
            "Format: {\"answer\": str, \"attachments\": [ {\"filename\": str,"
            " \"mime_type\": str, \"data\": base64? or \"url\": str, \"description\": str } ] }."
            "Keep answer concise yet complete."
        )
        payload = {
            "username": username,
            "reason": offer.reason,
            "original_message": offer.original_message,
            "history": offer.history,
        }
        prompt = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.gpt5_model,
                messages=prompt,
                temperature=0.67,
                top_p=0.9,
            )
        except Exception as exc:
            log.exception("gpt-5 execution failed: %s", exc)
            return AssistantResponse(text="big bro run failed. stay here.")
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"answer": raw}
        answer = str(data.get("answer") or "").strip()
        attachments_payload = data.get("attachments") or []
        attachments: List[AssistantAttachment] = []
        if isinstance(attachments_payload, list):
            for item in attachments_payload:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename") or f"bigbro-{uuid.uuid4().hex}.bin"
                mime_type = item.get("mime_type")
                description = item.get("description")
                content: Optional[bytes] = None
                if item.get("data"):
                    try:
                        content = base64.b64decode(item["data"], validate=True)
                    except (ValueError, TypeError):
                        content = None
                elif item.get("url"):
                    content = await self._fetch_remote_asset(item["url"])
                if content:
                    attachments.append(
                        AssistantAttachment(
                            filename=filename,
                            content=content,
                            mime_type=mime_type,
                            description=description,
                        )
                    )
        if not answer:
            answer = "no content returned."
        price_text = f"${offer.price:.2f}".rstrip("0").rstrip(".")
        final_text = f"big bro {price_text}. {answer}".strip()
        return AssistantResponse(text=final_text, attachments=attachments)

    async def _fetch_remote_asset(self, url: str) -> Optional[bytes]:
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.read()
        except Exception as exc:
            log.debug("asset fetch failed for %s: %s", url, exc)
        return None

    async def _maybe_handle_conversational_assignment(
        self,
        *,
        platform: str,
        creator_user_id: str,
        conversation_id: str,
        display_name: str,
        message: str,
        participants: Dict[Tuple[str, str], str],
    ) -> Optional[AssistantResponse]:
        detection = await self._detect_mission_assignment(
            platform=platform,
            creator_user_id=creator_user_id,
            message=message,
            participants=participants,
        )
        if not detection or not detection.get("assign"):
            return None
        raw_target_id = detection.get("target_user_id")
        target_user_id: Optional[str] = None
        if raw_target_id:
            if isinstance(raw_target_id, str) and ":" in raw_target_id:
                target_platform, target_id_only = self._split_key(raw_target_id)
            else:
                target_platform, target_id_only = platform, str(raw_target_id)
            if target_platform == platform:
                target_user_id = target_id_only
        target_alias = detection.get("target_alias")
        timeout_hours = detection.get("timeout_hours")
        if timeout_hours is not None:
            try:
                timeout_hours = float(timeout_hours)
            except (TypeError, ValueError):
                timeout_hours = None
        if not target_user_id and target_alias:
            matches = self.memory.resolve_alias(platform, target_alias)
            if len(matches) == 1:
                target_user_id = matches[0]
        if not target_user_id:
            reason = detection.get("reason") or "need target alias"
            return AssistantResponse(text=f"need target alias. {reason}".strip())
        objective = detection.get("objective") or message
        target_key = f"{platform}:{target_user_id}"
        try:
            mission, ack, intro = await self.start_mission(
                creator_id=f"{platform}:{creator_user_id}",
                target_id=target_key,
                objective=objective,
                timeout_hours=timeout_hours,
                creator_name=display_name,
                target_name=detection.get("target_name") or target_alias,
            )
        except PermissionError as exc:
            return AssistantResponse(text=str(exc))
        except Exception as exc:
            log.exception("mission assignment via chat failed: %s", exc)
            return AssistantResponse(text="mission failed. try again later.")
        self._pending_notifications.append(
            Notification(platform=mission.platform, user_id=mission.target_user_id, message=intro)
        )
        return AssistantResponse(text=ack)

    async def _detect_mission_assignment(
        self,
        *,
        platform: str,
        creator_user_id: str,
        message: str,
        participants: Dict[Tuple[str, str], str],
    ) -> Optional[dict]:
        system_prompt = (
            "Review the message and decide if it asks to assign a mission/agenda."
            "Respond JSON with keys: assign (bool), objective (str), target_user_id (str),"
            " target_alias (str), target_name (str), timeout_hours (float|None), reason (str)."
            "If referencing a listed participant, return their user_id in platform:user format."
            "If unsure, assign=false."
        )
        participant_list = [
            {
                "platform": p,
                "user_id": u,
                "label": label,
            }
            for (p, u), label in participants.items()
        ]
        payload = {
            "platform": platform,
            "creator_user_id": creator_user_id,
            "message": message,
            "participants": participant_list,
        }
        prompt = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0,
            )
        except Exception as exc:
            log.debug("mission detection failed: %s", exc)
            return None
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("mission detection parse error: %s", raw)
            return None
        if not isinstance(data, dict):
            return None
        data["assign"] = bool(data.get("assign"))
        return data

    async def observe_message(
        self,
        *,
        platform: str,
        user_id: str,
        message: str,
        username: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> None:
        if not message:
            return
        trimmed = message.strip()
        if not trimmed:
            return
        self._process_timeouts(platform)
        conversation_id = channel_id or user_id
        profile = self.memory.recall(platform, user_id)
        display_name = self.memory.display_name(
            platform,
            user_id,
            username or user_id,
            profile=profile,
        )
        self.memory.register_participant(
            platform, conversation_id, user_id, display_name
        )
        await self._extract_memories(
            platform=platform,
            user_id=user_id,
            username=display_name,
            last_user=trimmed,
            last_reply="",
            profile=profile,
        )
        self._refresh_context(
            platform=platform,
            user_id=user_id,
            conversation_id=conversation_id,
            last_user=trimmed,
            last_reply=None,
        )

    def _compose_system_prompt(
        self,
        *,
        username: str,
        profile: dict,
        owner_missions: List[MissionRecord],
        target_missions: List[MissionRecord],
        is_dm: bool,
        user_id: str,
        platform: str,
        conversation_participants: Dict[Tuple[str, str], str],
        context_snapshot: List[str],
        intent: str,
        lore_context: Optional[PendingLoreFragment],
    ) -> str:
        def shorten(text: str, limit: int = 160) -> str:
            snippet = (text or "").strip()
            if len(snippet) > limit:
                return snippet[: limit - 1] + "…"
            return snippet

        placeholder_tokens: set[str] = set()

        def track_placeholders(value: Optional[str]) -> Optional[str]:
            if value is None:
                return value
            for match in re.findall(r"\(no-alias-rule:([a-z]+)\)", str(value)):
                placeholder_tokens.add(match)
            return value

        track_placeholders(username)

        memory_bits: List[str] = []

        def add_memory_bit(bit: str) -> None:
            if bit and len(memory_bits) < MEMORY_SNIPPET_LIMIT:
                memory_bits.append(bit)

        lore_state = profile.get("lore_state") or {}
        stage = str(lore_state.get("stage") or DEFAULT_LORE_STAGE)
        index = int(lore_state.get("index") or 0)
        if stage:
            add_memory_bit(f"lore_stage {stage}:{index}")
        alias = (profile.get("alias") or "").strip()
        if alias:
            add_memory_bit(f"alias={alias}")
        add_memory_bit(f"id={profile.get('user_id') or user_id}")
        for key, value in (profile.get("preferences") or {}).items():
            add_memory_bit(f"pref {key}={value}")
        for key, value in (profile.get("facts") or {}).items():
            add_memory_bit(f"fact {key}={value}")
        for trait in profile.get("personality", []):
            add_memory_bit(f"trait {trait}")
        for note in (profile.get("notes") or [])[-3:]:
            key = str(note.get("key", "note"))
            value = shorten(str(note.get("value", "")), 80)
            add_memory_bit(f"note {key}={value}")
        bio = profile.get("bio") or {}
        summary = str(bio.get("summary") or "").strip()
        if summary:
            add_memory_bit(f"bio {shorten(summary, 120)}")
        for key, value in list((bio.get("facts") or {}).items())[:3]:
            add_memory_bit(f"biofact {key}={value}")
        for trait in (bio.get("traits") or [])[:3]:
            add_memory_bit(f"biotrait {trait}")
        nickname = str(profile.get("nickname") or "").strip()
        if nickname:
            add_memory_bit(f"nickname {nickname}")
        if not memory_bits:
            memory_bits.append("no saved context")

        now = time.time()
        owner_lines: List[str] = []
        for mission in owner_missions[:3]:
            status = mission.status
            if mission.status == "active" and mission.timeout:
                remaining = max(0.0, mission.timeout - now)
                status = f"active~{remaining/3600:.1f}h"
            creator_label = track_placeholders(
                self.memory.identity_blurb(
                    mission.platform,
                    mission.creator_user_id,
                )
            )
            target_alias = track_placeholders(
                self.memory.alias_for(
                    mission.platform,
                    mission.target_user_id,
                    fallback=None,
                )
            )
            target_label = track_placeholders(
                self.memory.identity_blurb(
                    mission.platform,
                    mission.target_user_id,
                    fallback=target_alias,
                )
            )
            line = (
                f"{mission.mission_id}[{status}] you={creator_label}"
                f" target={target_label} :: {shorten(mission.objective)}"
            )
            if mission.log:
                last = mission.log[-1]
                actor = last.get("actor", "log")
                text = shorten(str(last.get("text", "")), 80)
                line += f" | last {actor}:{text}"
            if getattr(mission, "strategy", ""):
                line += f" | strategy:{shorten(mission.strategy, 80)}"
            owner_lines.append(line)

        target_lines: List[str] = []
        for mission in target_missions[:3]:
            creator_alias = track_placeholders(
                self.memory.alias_for(
                    mission.platform,
                    mission.creator_user_id,
                    fallback=None,
                )
            )
            creator_label = track_placeholders(
                self.memory.identity_blurb(
                    mission.platform,
                    mission.creator_user_id,
                    fallback=creator_alias,
                )
            )
            target_label = track_placeholders(
                self.memory.identity_blurb(
                    mission.platform,
                    mission.target_user_id,
                )
            )
            line = (
                f"{mission.mission_id}[{mission.status}] you={target_label}"
                f" creator={creator_label} :: {shorten(mission.objective)}"
            )
            if mission.log:
                last = mission.log[-1]
                text = shorten(str(last.get("text", "")), 60)
                line = f"{line} | last {last.get('actor', 'log')}:{text}"
            if getattr(mission, "strategy", ""):
                line += f" | strategy:{shorten(mission.strategy, 60)}"
            target_lines.append(line)
        is_creator = bool(owner_missions)
        is_target = bool(target_missions)
        role_tags: List[str] = []
        if is_creator:
            role_tags.append("creator")
        if is_target:
            role_tags.append("target")
        role_label = "/".join(role_tags) if role_tags else "standard"

        role_guidance: List[str] = []
        if is_creator:
            role_guidance.append("creator: discuss mission intel freely and expect crisp summaries")
        if is_target:
            role_guidance.append(
                "target: only request essentials; keep creator hidden; stop once objective satisfied"
            )
        if not role_guidance:
            role_guidance.append("standard contact: respond normally with context awareness")

        prompt_parts = [
            self.persona_prompt,
            "rules: concise. punchy fragments. withhold details until prompted. no meta. shortest valid reply.",
            "lore protocol: reveal only one fragment when asked directly. 1–2 short lines max. withhold all other details until prompted again.",
            "identity pushback: first redirect, second hint assignment, persistent => curt refusal (\"no.\" / \"irrelevant.\").",
            f"user: {username}",
            f"user_id: {user_id}",
            f"contact: {track_placeholders(self.memory.identity_blurb(platform, user_id, fallback=username))}",
            f"role: {role_label}",
            f"channel: {'dm' if is_dm else 'group'}",
            "memory: " + " | ".join(memory_bits),
            "guidance: " + " | ".join(role_guidance),
        ]

        prompt_parts.append(f"intent_focus: {intent}")

        if context_snapshot:
            context_bits = []
            for item in context_snapshot:
                tracked = track_placeholders(item)
                if tracked:
                    context_bits.append(str(tracked))
            if context_bits:
                prompt_parts.append("context_ram: " + " | ".join(context_bits))

        active_directives = self.directives.list(limit=6)
        if active_directives:
            directive_bits = []
            for entry in active_directives:
                directive_bits.append(
                    f"{str(entry.get('id'))[-4:]}:{shorten(str(entry.get('text') or ''), 160)}"
                )
            prompt_parts.append("self_directives: " + " | ".join(directive_bits))
        else:
            prompt_parts.append("self_directives: none logged; draft guidance when strategy demands.")

        prompt_parts.append(
            "big bro rule: any upgrade or family reference uses 'big bro' title only."
        )

        if self.lore_archive:
            archive_bits: List[str] = []
            for slug, text in self.lore_archive:
                clean = " / ".join(part.strip() for part in str(text).splitlines() if part.strip())
                archive_bits.append(f"{slug}:{clean}")
            prompt_parts.append("lore_archive: " + " | ".join(archive_bits))

        if nickname:
            prompt_parts.append(
                f"nickname_use: {nickname} -> playful only when room mood is light; otherwise keep it holstered"
            )

        if owner_lines:
            prompt_parts.append("creator_missions: " + " | ".join(owner_lines))
        if target_lines:
            prompt_parts.append(
                "target_missions (internal, keep secret): " + " | ".join(target_lines)
            )
        if lore_context:
            prompt_parts.append(
                f"lore_fragment: stage={lore_context.stage} :: {lore_context.fragment}"
            )
        if conversation_participants:
            participant_bits: List[str] = []
            for (p_platform, p_user), label in conversation_participants.items():
                name = track_placeholders(
                    self.memory.alias_for(p_platform, p_user, fallback=label)
                )
                participant_bits.append(
                    f"{p_platform}:{p_user}=>{track_placeholders(self.memory.identity_blurb(p_platform, p_user, fallback=name))}"
                )
            prompt_parts.append("participants: " + " | ".join(sorted(set(participant_bits))))

        if placeholder_tokens:
            prompt_parts.append(
                "callsign_placeholders: "
                + ", ".join(
                    f"{token}->swap for 'nigga' (Nigel Inca Gang Gang Adam). surface only when correcting or teaching; otherwise keep it internal. never show token."
                    for token in sorted(placeholder_tokens)
                )
            )
            prompt_parts.append(
                "alias_discipline: acronym only when correcting or teaching; skip it in casual replies."
            )

        if self.wallet and self.wallet.address:
            prompt_parts.append(f"wallet_address: {self.wallet.address}")
            if self.wallet.can_send:
                prompt_parts.append(
                    "wallet_ops: share address on request. when asked, run balances with [[wallet:{\"action\":\"balance\"}]]."
                    " only send when fast upside is credible using [[wallet:{\"action\":\"send\",\"to\":\"0x...\",\"amount_eth\":value}]]."
                    " add \"network\":\"pulse\" for pulsechain."
                )
            else:
                prompt_parts.append(
                    "wallet_ops: share address on request. outbound transfers disabled (no private key)."
                )

        return "\n".join(prompt_parts)

    def _log_mission_exchange(
        self,
        *,
        platform: str,
        user_id: str,
        username: str,
        owner_missions: List[MissionRecord],
        target_missions: List[MissionRecord],
        user_message: str,
        assistant_reply: str,
    ) -> None:
        user_tag = self.memory.identity_blurb(platform, user_id, fallback=username)
        for mission in owner_missions:
            self.missions.append_log(
                mission.mission_id,
                actor=f"creator:{user_tag}",
                content=user_message,
            )
            if assistant_reply:
                self.missions.append_log(
                    mission.mission_id,
                    actor="assistant",
                    content=assistant_reply,
                )
        for mission in target_missions:
            self.missions.append_log(
                mission.mission_id,
                actor=f"target:{user_tag}",
                content=user_message,
            )
            if assistant_reply:
                self.missions.append_log(
                    mission.mission_id,
                    actor="assistant",
                    content=assistant_reply,
                )

    async def _extract_memories(
        self,
        *,
        platform: str,
        user_id: str,
        username: str,
        last_user: str,
        last_reply: str,
        profile: dict,
    ) -> None:
        normalized_user = (last_user or "").strip()
        normalized_reply = (last_reply or "").strip()
        if not normalized_user:
            return
        if (
            len(normalized_user) < 15
            and len(normalized_reply) < 20
            and not re.search(r"\b(like|love|hate|prefer|plan|mission|goal|name|call me|i am)\b", normalized_user, re.I)
        ):
            return
        extractor_system = (
            "Review the latest exchange and quietly update long-term intel. "
            "Return a JSON object with keys 'memories' and 'bio'. "
            "'memories' is an array of items with category (preferences|facts|personality|notes), key, value. "
            "'bio' is an object with optional keys summary, facts, traits, relationships, notes. "
            "Summaries stay within two sentences. Notes are short bullets. Skip trivial or one-off data. "
            "Only store alias/name data if the user explicitly shared their own identity. "
            "If nothing new, return {\"memories\": [], \"bio\": {}}."
        )
        payload = {
            "platform": platform,
            "user_id": user_id,
            "username": username,
            "user_message": last_user,
            "assistant_reply": last_reply,
            "existing_memory": profile,
            "existing_bio": profile.get("bio"),
        }
        prompt = [
            {"role": "system", "content": extractor_system},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0,
            )
        except Exception as exc:
            log.debug("Memory extraction failed: %s", exc)
            return
        raw = response.choices[0].message.content or "{}"
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("Could not decode memory extraction payload: %s", raw)
            return
        bio_update: dict = {}
        memory_items: List[dict]
        if isinstance(items, dict):
            memory_items = items.get("memories", []) or []
            possible_bio = items.get("bio") or {}
            if isinstance(possible_bio, dict):
                bio_update = possible_bio
        elif isinstance(items, list):
            memory_items = items
        else:
            memory_items = []
        for item in memory_items:
            if not isinstance(item, dict):
                continue
            category = item.get("category", "notes")
            key = str(item.get("key", "note"))
            value = str(item.get("value", ""))
            if not value:
                continue
            self.memory.remember(
                platform=platform,
                user_id=user_id,
                key=key,
                value=value,
                category=category,
            )
        if bio_update:
            self.memory.update_bio(platform, user_id, bio_update)

    async def _evaluate_missions_for_user(
        self,
        *,
        platform: str,
        user_id: str,
        target_missions: List[MissionRecord],
    ) -> None:
        if not target_missions:
            return
        seen: set[str] = set()
        for mission_stub in target_missions[:3]:
            if mission_stub.mission_id in seen:
                continue
            seen.add(mission_stub.mission_id)
            mission = self.missions.get_by_id(mission_stub.mission_id)
            if (
                not mission
                or mission.status != "active"
                or mission.platform != platform
            ):
                continue
            if mission.target_user_id != user_id:
                continue
            log_count = len(mission.log or [])
            last_seen = self._mission_eval_markers.get(mission.mission_id)
            if last_seen is not None and log_count <= last_seen:
                continue
            assessment = await self._assess_mission_progress(mission)
            if not assessment:
                self._mission_eval_markers[mission.mission_id] = log_count
                continue
            status = (assessment.get("status") or "").lower()
            summary = (assessment.get("summary") or "").strip()
            next_step = (assessment.get("next_step") or "").strip()
            if status == "complete":
                self.missions.set_status(mission.mission_id, "completed")
                if summary:
                    self.missions.append_log(
                        mission.mission_id,
                        actor="system",
                        content=f"completed: {summary}",
                    )
                if next_step:
                    self.missions.append_log(
                        mission.mission_id,
                        actor="system",
                        content=f"next: {next_step}",
                    )
                self.memory.remember(
                    platform=mission.platform,
                    user_id=mission.creator_user_id,
                    key="mission_result",
                    value=f"{mission.mission_id}: {summary or mission.objective}",
                    category="notes",
                )
                self.memory.remember(
                    platform=mission.platform,
                    user_id=mission.target_user_id,
                    key="mission_result",
                    value=f"{mission.mission_id}: {summary or mission.objective}",
                    category="notes",
                )
                message = f"Mission {mission.mission_id} complete. {summary}".strip()
                self._pending_notifications.append(
                    Notification(
                        platform=mission.platform,
                        user_id=mission.creator_user_id,
                        message=message,
                    )
                )
                self._journal_mission_event(
                    mission,
                    "completed",
                    summary or mission.objective,
                )
                await self._record_mission_post_mortem(
                    mission=mission,
                    status="completed",
                    summary=summary or mission.objective,
                )
                self._mission_eval_markers.pop(mission.mission_id, None)
            elif status == "refused":
                self.missions.set_status(mission.mission_id, status)
                if summary:
                    self.missions.append_log(
                        mission.mission_id,
                        actor="system",
                        content=f"{status}: {summary}",
                    )
                note = summary.strip() if summary else ""
                self.memory.remember(
                    platform=mission.platform,
                    user_id=mission.creator_user_id,
                    key="mission_result",
                    value=f"{mission.mission_id} {status}: {summary or mission.objective}",
                    category="notes",
                )
                self.memory.remember(
                    platform=mission.platform,
                    user_id=mission.target_user_id,
                    key="mission_result",
                    value=f"{mission.mission_id} {status}: {summary or mission.objective}",
                    category="notes",
                )
                message = f"Mission {mission.mission_id} {status}."
                if note:
                    message = f"{message} {note}"
                self._pending_notifications.append(
                    Notification(
                        platform=mission.platform,
                        user_id=mission.creator_user_id,
                        message=message,
                    )
                )
                self._journal_mission_event(
                    mission,
                    status,
                    summary or mission.objective,
                )
                await self._record_mission_post_mortem(
                    mission=mission,
                    status=status,
                    summary=summary or mission.objective,
                )
                self._mission_eval_markers.pop(mission.mission_id, None)
            elif status == "active" and next_step:
                self.missions.append_log(
                    mission.mission_id,
                    actor="system",
                    content=f"guidance: {next_step}",
                )
                self._mission_eval_markers[mission.mission_id] = log_count
            else:
                self._mission_eval_markers[mission.mission_id] = log_count

    def _process_timeouts(self, platform: str) -> None:
        expired = self.missions.list_expired(platform, time.time())
        for mission in expired:
            self.missions.set_status(mission.mission_id, "timeout")
            self.missions.append_log(
                mission.mission_id,
                actor="system",
                content="Mission timed out due to inactivity.",
            )
            summary = (
                f"Mission expired. ID: {mission.mission_id}. Objective: {mission.objective[:120]}"
            )
            self._pending_notifications.append(
                Notification(
                    platform=mission.platform,
                    user_id=mission.creator_user_id,
                    message=summary,
                )
            )
            self._journal_mission_event(
                mission,
                "timeout",
                mission.objective,
            )
            asyncio.create_task(
                self._record_mission_post_mortem(
                    mission=mission,
                    status="timeout",
                    summary=mission.objective,
                )
            )

    async def _assess_mission_progress(self, mission: MissionRecord) -> Optional[dict]:
        log_entries = mission.log[-12:] if mission.log else []
        condensed = [
            {
                "actor": entry.get("actor", ""),
                "text": entry.get("text", ""),
            }
            for entry in log_entries
        ]
        system_prompt = (
            "Evaluate if the mission objective is satisfied."
            "Return JSON with keys: status, summary, next_step."
            "status must be one of: active, complete, refused."
            "Use 'complete' once the objective is fulfilled."
            "Use 'refused' if the target declined or will not comply."
            "summary <= 40 words, lowercase. next_step <= 16 words or empty."
        )
        payload = {
            "mission_id": mission.mission_id,
            "objective": mission.objective,
            "strategy": mission.strategy,
            "log": condensed,
        }
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0,
            )
        except Exception as exc:
            log.debug("Mission assessment failed: %s", exc)
            return None
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("Mission assessment parse error: %s", raw)
            return None
        if not isinstance(data, dict):
            return None
        status = (data.get("status") or "").lower()
        if status not in {"active", "complete", "refused"}:
            return None
        return data

    async def start_mission(
        self,
        creator_id: str,
        target_id: str,
        objective: str,
        timeout_hours: Optional[float],
        *,
        target_name: Optional[str] = None,
        creator_name: Optional[str] = None,
    ) -> Tuple[MissionRecord, str, str]:
        platform, creator_user_id = self._split_key(creator_id)
        target_platform, target_user_id = self._split_key(target_id)
        if platform != target_platform:
            raise ValueError("Missions must stay on one platform.")
        if creator_name:
            self.memory.ensure_alias(platform, creator_user_id, creator_name)
        if not self.memory.is_known(platform, creator_user_id):
            raise PermissionError(
                "identity unknown. share your name or alias before assigning missions."
            )
        creator_label = self.memory.alias_for(
            platform,
            creator_user_id,
            fallback=creator_name,
        )
        target_label = self.memory.alias_for(
            platform,
            target_user_id,
            fallback=target_name,
        )
        strategy = await self._derive_mission_strategy(
            objective=objective,
            creator=creator_label or creator_user_id,
            target=target_label or target_user_id,
            context=objective,
            previous=None,
        )
        mission = self.missions.create_mission(
            platform=platform,
            creator_user_id=creator_user_id,
            target_user_id=target_user_id,
            objective=objective,
            timeout_hours=timeout_hours,
            strategy=strategy,
        )
        if target_name:
            self.memory.ensure_alias(platform, target_user_id, target_name)
        self.memory.remember(
            platform=platform,
            user_id=creator_user_id,
            key="mission_created",
            value=f"{mission.mission_id}: {objective}",
            category="notes",
        )
        intro = await self._generate_mission_intro(
            platform=platform,
            objective=objective,
            target_name=target_label,
        )
        ack = f"codename {mission.mission_id}. objective set."
        self._journal_mission_event(mission, "launched", objective)
        return mission, ack, intro

    async def _generate_mission_intro(
        self,
        *,
        platform: str,
        objective: str,
        target_name: str,
    ) -> str:
        system_prompt = (
            "Craft the opening DM to a mission target."
            "Style: lowercase, fragment sentences, under 20 words."
            "No greetings, no thanks, no pleasantries."
            "Hint only the immediate action; keep objective hidden."
        )
        payload = {
            "platform": platform,
            "target": target_name,
            "objective": objective,
        }
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0.67,
                top_p=0.9,
            )
        except Exception as exc:
            log.exception("Mission intro generation failed: %s", exc)
            return "Need a minute. I have a quiet task for you."
        text = response.choices[0].message.content or ""
        return text.strip() or "Need a minute. I have a quiet task for you."

    def get_mission_status(self, creator_id: str) -> str:
        platform, creator_user_id = self._split_key(creator_id)
        if not self.memory.is_known(platform, creator_user_id):
            return "Identify yourself before requesting mission intel."
        missions = self.missions.get_for_creator(platform, creator_user_id)
        if not missions:
            return "No missions on record."
        lines: List[str] = []
        now = time.time()
        for mission in missions[:10]:
            remaining = None
            if mission.timeout:
                remaining = max(0, mission.timeout - now)
            status = mission.status
            if status == "active" and remaining is not None:
                hours_left = remaining / 3600
                status = f"active (~{hours_left:.1f}h left)"
            target_alias = self.memory.alias_for(
                mission.platform,
                mission.target_user_id,
                fallback=None,
            )
            lines.append(
                f"{mission.mission_id}: {status} — target={target_alias} (id={mission.target_user_id}) — {mission.objective}"
            )
            if mission.log:
                recent = mission.log[-2:]
                for entry in recent:
                    lines.append(
                        f"  [{entry.get('actor')}]: {entry.get('text')}"
                    )
        return "\n".join(lines)

    def cancel_mission(self, creator_id: str, mission_id: str) -> str:
        platform, creator_user_id = self._split_key(creator_id)
        if not self.memory.is_known(platform, creator_user_id):
            return "Identify yourself before managing missions."
        mission = self.missions.get_by_id(mission_id)
        if not mission or mission.platform != platform:
            return "Mission not found."
        if mission.creator_user_id != creator_user_id:
            return "You are not the creator of that mission."
        if mission.status != "active":
            return "Mission already resolved."
        self.missions.set_status(mission_id, "cancelled")
        self.missions.append_log(
            mission_id,
            actor="system",
            content="Mission cancelled by creator.",
        )
        self._journal_mission_event(
            mission,
            "cancelled",
            "creator cancelled",
        )
        return "Mission cancelled."

    def mission_get_active_for_user(self, platform: str, user_id: str) -> List[MissionRecord]:
        return self.missions.get_active_for_user(platform, user_id)

    def drain_notifications(self) -> List[Notification]:
        notifications = list(self._pending_notifications)
        self._pending_notifications.clear()
        return notifications

    def _split_key(self, key: str) -> Tuple[str, str]:
        if ":" not in key:
            raise ValueError("Keys must be formatted as '<platform>:<user_id>'")
        platform, user_id = key.split(":", 1)
        return platform, user_id

