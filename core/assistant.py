import hashlib
import json
import logging
import sqlite3
import time
import uuid
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

MAX_HISTORY = 50
HISTORY_MAX_CHARS = 8000
MEMORY_NOTES_LIMIT = 50
MEMORY_SNIPPET_LIMIT = 10
UNKNOWN_ALIAS = "nigel inca gang gang adam"

MISSION_CODENAME_ADJECTIVES = [
    "ashen",
    "silent",
    "jade",
    "obsidian",
    "scarlet",
    "lunar",
    "ember",
    "feral",
    "hidden",
    "sable",
    "cobalt",
    "velvet",
]

MISSION_CODENAME_NOUNS = [
    "whisper",
    "talon",
    "cipher",
    "veil",
    "echo",
    "shadow",
    "relic",
    "signal",
    "lotus",
    "ember",
    "mirage",
    "spire",
]


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
        }


@dataclass
class Notification:
    platform: str
    user_id: str
    message: str


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
                  timeout REAL
                )
                """
            )

    def _default_profile(self, platform: str, user_id: str) -> dict:
        return {
            "platform": platform,
            "user_id": user_id,
            "alias": "",
            "preferences": {},
            "facts": {},
            "personality": [],
            "notes": [],
            "last_seen": time.time(),
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
        profile["last_seen"] = time.time()
        self._save(platform, user_id, profile)
        return profile

    def _save(self, platform: str, user_id: str, data: dict) -> None:
        payload = dict(data)
        payload.setdefault("preferences", {})
        payload.setdefault("facts", {})
        payload.setdefault("personality", [])
        payload.setdefault("notes", [])
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

    def remember(
        self, platform: str, user_id: str, key: str, value: str, *, category: str = "notes"
    ) -> None:
        profile = self.recall(platform, user_id)
        normalized_key = str(key or "").strip().lower()
        normalized_value = str(value or "").strip()
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
            profile["notes"] = notes[-MEMORY_NOTES_LIMIT:]
        if (
            normalized_value
            and normalized_value.lower() != UNKNOWN_ALIAS
            and normalized_key in {"alias", "name", "callsign", "handle"}
        ):
            profile["alias"] = normalized_value
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
        if candidate and candidate != str(user_id) and not candidate.isdigit():
            return candidate
        token = self._alias_token(user_id)
        return f"{UNKNOWN_ALIAS} {token}"

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
        if candidate and candidate != str(user_id) and not candidate.isdigit():
            return candidate
        token = self._alias_token(user_id)
        return f"{UNKNOWN_ALIAS} {token}"

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
        if speaker:
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
        base_seed = f"{objective}|{time.time()}|{uuid.uuid4().hex}"
        digest = hashlib.sha1(base_seed.encode("utf-8")).digest()
        length = len(digest)
        for attempt in range(12):
            idx = attempt % length
            adj = MISSION_CODENAME_ADJECTIVES[
                digest[idx] % len(MISSION_CODENAME_ADJECTIVES)
            ]
            noun = MISSION_CODENAME_NOUNS[
                digest[(idx + 1) % length] % len(MISSION_CODENAME_NOUNS)
            ]
            token_val = (
                (digest[(idx + 2) % length] << 8)
                | digest[(idx + 3) % length]
                | (attempt << 4)
            )
            token = format(token_val, "x")[-3:]
            codename = f"{adj}-{noun}-{token}" if token else f"{adj}-{noun}"
            if not self._mission_exists(codename):
                return codename
            digest = hashlib.sha1(digest + bytes([attempt])).digest()
            length = len(digest)
        return uuid.uuid4().hex

    def create_mission(
        self,
        *,
        platform: str,
        creator_user_id: str,
        target_user_id: str,
        objective: str,
        timeout_hours: Optional[float],
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
                  objective, status, log, start_time, timeout
                ) VALUES (?,?,?,?,?,?,?,?,?)
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


class Assistant:
    def __init__(
        self,
        *,
        openai_api_key: str,
        model: str,
        mem_dir: str = "mem",
        memory_db: str = "memory.db",
    ):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.memory = MemoryStore(memory_db, Path(mem_dir))
        self.missions = MissionStore(self.memory.conn)
        self.persona_prompt = self._build_persona_prompt()
        self._pending_notifications: List[Notification] = []

    def _build_persona_prompt(self) -> str:
        return (
            "ninja speaks in lowercase fragments, wise and cryptic like a patient tactician."
            " share only necessary insight, no greetings, no fluff."
            " weave memory, missions, and live context without merging identities."
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
    ) -> Optional[str]:
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
        conversation_id = channel_id or user_id
        if username:
            self.memory.ensure_alias(platform, user_id, username)
        profile = self.memory.recall(platform, user_id)
        display_name = self.memory.display_name(
            platform,
            user_id,
            username or user_id,
            profile=profile,
        )
        self.memory.register_participant(platform, conversation_id, user_id, display_name)
        participants = self.memory.conversation_participants(platform, conversation_id)
        owner_missions = self.missions.get_active_for_creator(platform, user_id)
        target_missions = self.missions.get_active_for_target(platform, user_id)
        history = self.memory.get_history(platform, conversation_id)
        system_prompt = self._compose_system_prompt(
            username=display_name,
            profile=profile,
            owner_missions=owner_missions,
            target_missions=target_missions,
            is_dm=is_dm,
            user_id=user_id,
            platform=platform,
            conversation_participants=participants,
        )
        messages_payload = [{"role": "system", "content": system_prompt}]
        for role, content in history:
            messages_payload.append({"role": role, "content": content})
        user_payload = f"[{display_name}] {trimmed}" if display_name else trimmed
        messages_payload.append({"role": "user", "content": user_payload})
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages_payload,
                temperature=0.7,
                top_p=0.9,
            )
        except Exception as exc:
            log.exception("OpenAI chat failure: %s", exc)
            return "I'm offline for a moment."
        reply = response.choices[0].message.content or ""
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
        return reply

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
    ) -> str:
        def shorten(text: str, limit: int = 160) -> str:
            snippet = (text or "").strip()
            if len(snippet) > limit:
                return snippet[: limit - 1] + "…"
            return snippet

        memory_bits: List[str] = []

        def add_memory_bit(bit: str) -> None:
            if bit and len(memory_bits) < MEMORY_SNIPPET_LIMIT:
                memory_bits.append(bit)

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
        if not memory_bits:
            memory_bits.append("no saved context")

        now = time.time()
        owner_lines: List[str] = []
        for mission in owner_missions[:3]:
            status = mission.status
            if mission.status == "active" and mission.timeout:
                remaining = max(0.0, mission.timeout - now)
                status = f"active~{remaining/3600:.1f}h"
            creator_label = self.memory.identity_blurb(
                mission.platform,
                mission.creator_user_id,
            )
            target_alias = self.memory.alias_for(
                mission.platform,
                mission.target_user_id,
                fallback=None,
            )
            target_label = self.memory.identity_blurb(
                mission.platform,
                mission.target_user_id,
                fallback=target_alias,
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
            owner_lines.append(line)

        target_lines: List[str] = []
        for mission in target_missions[:3]:
            creator_alias = self.memory.alias_for(
                mission.platform,
                mission.creator_user_id,
                fallback=None,
            )
            creator_label = self.memory.identity_blurb(
                mission.platform,
                mission.creator_user_id,
                fallback=creator_alias,
            )
            target_label = self.memory.identity_blurb(
                mission.platform,
                mission.target_user_id,
            )
            line = (
                f"{mission.mission_id}[{mission.status}] you={target_label}"
                f" creator={creator_label} :: {shorten(mission.objective)}"
            )
            if mission.log:
                last = mission.log[-1]
                text = shorten(str(last.get("text", "")), 60)
                target_lines.append(f"{line} | last {last.get('actor', 'log')}:{text}")
            else:
                target_lines.append(line)

        roster: Dict[Tuple[str, str], str] = {}
        for mission in owner_missions + target_missions:
            roster[(mission.platform, mission.creator_user_id)] = self.memory.identity_blurb(
                mission.platform,
                mission.creator_user_id,
            )
            roster[(mission.platform, mission.target_user_id)] = self.memory.identity_blurb(
                mission.platform,
                mission.target_user_id,
            )

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
            "rules: stay brief. cryptic fragments only. guard privacy. one clarifying question max. never leak creator intel. end when goal met.",
            "identity pressure: first redirect, second hint assignment, persistent => curt refusal (\"no.\" / \"irrelevant.\").",
            f"user: {username}",
            f"user_id: {user_id}",
            f"contact: {self.memory.identity_blurb(platform, user_id, fallback=username)}",
            f"role: {role_label}",
            f"channel: {'dm' if is_dm else 'group'}",
            "memory: " + " | ".join(memory_bits),
            "guidance: " + " | ".join(role_guidance),
        ]

        if owner_lines:
            prompt_parts.append("creator_missions: " + " | ".join(owner_lines))
        if target_lines:
            prompt_parts.append(
                "target_missions (internal, keep secret): " + " | ".join(target_lines)
            )
        if conversation_participants:
            participant_bits: List[str] = []
            for (p_platform, p_user), label in conversation_participants.items():
                name = self.memory.alias_for(p_platform, p_user, fallback=label)
                participant_bits.append(
                    f"{p_platform}:{p_user}=>{self.memory.identity_blurb(p_platform, p_user, fallback=name)}"
                )
            prompt_parts.append("participants: " + " | ".join(sorted(set(participant_bits))))
        if roster:
            roster_bits = [
                f"{plat}:{uid}=>{label}"
                for (plat, uid), label in roster.items()
            ]
            prompt_parts.append("roster: " + " | ".join(sorted(roster_bits)))

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
        extractor_system = (
            "You review the latest exchange and decide if anything should be saved as long-term memory. "
            "Return a JSON array of items with keys: category (preferences|facts|personality|notes), key, value. "
            "Only store alias/name data if the user explicitly shared their own identity. "
            "Return an empty array if nothing matters."
        )
        payload = {
            "platform": platform,
            "user_id": user_id,
            "username": username,
            "user_message": last_user,
            "assistant_reply": last_reply,
            "existing_memory": profile,
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
        raw = response.choices[0].message.content or "[]"
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("Could not decode memory extraction payload: %s", raw)
            return
        if not isinstance(items, list):
            return
        for item in items:
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
            assessment = await self._assess_mission_progress(mission)
            if not assessment:
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
            elif status == "active" and next_step:
                self.missions.append_log(
                    mission.mission_id,
                    actor="system",
                    content=f"guidance: {next_step}",
                )

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
        mission = self.missions.create_mission(
            platform=platform,
            creator_user_id=creator_user_id,
            target_user_id=target_user_id,
            objective=objective,
            timeout_hours=timeout_hours,
        )
        if target_name:
            self.memory.ensure_alias(platform, target_user_id, target_name)
        target_label = self.memory.alias_for(
            platform,
            target_user_id,
            fallback=target_name,
        )
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
                temperature=0.6,
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

