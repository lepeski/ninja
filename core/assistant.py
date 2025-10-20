import json
import logging
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

MAX_HISTORY = 50
HISTORY_MAX_CHARS = 8000
MEMORY_NOTES_LIMIT = 50


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
        self._init_db()
        self._history: Dict[Tuple[str, str], Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY)
        )
        self._seen_first_contact: set[Tuple[str, str]] = set()

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

    def remember(
        self, platform: str, user_id: str, key: str, value: str, *, category: str = "notes"
    ) -> None:
        profile = self.recall(platform, user_id)
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
        self._save(platform, user_id, profile)

    def log_history(self, platform: str, conversation_id: str, role: str, content: str) -> None:
        key = (platform, conversation_id)
        history = self._history[key]
        history.append((role, content))
        total_chars = sum(len(item[1]) for item in history)
        while total_chars > HISTORY_MAX_CHARS and len(history) > 1:
            removed = history.popleft()
            total_chars -= len(removed[1])

    def get_history(self, platform: str, conversation_id: str) -> List[Tuple[str, str]]:
        return list(self._history[(platform, conversation_id)])

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

    def create_mission(
        self,
        *,
        platform: str,
        creator_user_id: str,
        target_user_id: str,
        objective: str,
        timeout_hours: Optional[float],
    ) -> MissionRecord:
        mission_id = uuid.uuid4().hex
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
            "You are Ninja, a focused multi-platform agent. Maintain a concise, calm, and capable tone. "
            "Deliver only useful information. Avoid filler and forced theatrics. Ask a clarifying question only when it is required to proceed. "
            "Blend human intuition with professionalism, keep context, and protect mission privacy."
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
        profile = self.memory.recall(platform, user_id)
        owner_missions = self.missions.get_active_for_creator(platform, user_id)
        target_missions = self.missions.get_active_for_target(platform, user_id)
        history = self.memory.get_history(platform, conversation_id)
        system_prompt = self._compose_system_prompt(
            username=username or user_id,
            profile=profile,
            owner_missions=owner_missions,
            target_missions=target_missions,
            is_dm=is_dm,
        )
        messages_payload = [{"role": "system", "content": system_prompt}]
        for role, content in history:
            messages_payload.append({"role": role, "content": content})
        messages_payload.append({"role": "user", "content": trimmed})
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
        self.memory.log_history(platform, conversation_id, "user", trimmed)
        if reply:
            self.memory.log_history(platform, conversation_id, "assistant", reply)
        self._log_mission_exchange(
            platform=platform,
            user_id=user_id,
            username=username or user_id,
            owner_missions=owner_missions,
            target_missions=target_missions,
            user_message=trimmed,
            assistant_reply=reply,
        )
        await self._extract_memories(
            platform=platform,
            user_id=user_id,
            username=username or user_id,
            last_user=trimmed,
            last_reply=reply,
            profile=profile,
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
    ) -> str:
        memory_lines: List[str] = []
        prefs = profile.get("preferences") or {}
        for key, value in prefs.items():
            memory_lines.append(f"Preference - {key}: {value}")
        facts = profile.get("facts") or {}
        for key, value in facts.items():
            memory_lines.append(f"Fact - {key}: {value}")
        for trait in profile.get("personality", []):
            memory_lines.append(f"Personality note: {trait}")
        for note in profile.get("notes", [])[-5:]:
            key = note.get("key", "note")
            value = note.get("value", "")
            memory_lines.append(f"Recent note ({key}): {value}")
        if not memory_lines:
            memory_lines.append("No stored personal details yet.")

        owner_lines: List[str] = []
        for mission in owner_missions:
            owner_lines.append(
                f"Mission {mission.mission_id}: {mission.objective} (status: {mission.status})"
            )
            if mission.log:
                recent = mission.log[-3:]
                for entry in recent:
                    owner_lines.append(
                        f"  log[{entry.get('actor')}]: {entry.get('text')}"
                    )
        if not owner_lines:
            owner_lines.append("No creator missions for this user.")

        target_lines: List[str] = []
        for mission in target_missions:
            target_lines.append(
                f"Mission {mission.mission_id} objective (keep private): {mission.objective}"
            )
            if mission.log:
                recent = mission.log[-3:]
                for entry in recent:
                    target_lines.append(
                        f"  log[{entry.get('actor')}]: {entry.get('text')}"
                    )
        if not target_lines:
            target_lines.append("No active missions targeting this user.")

        prompt_parts = [
            self.persona_prompt,
            "Context: you are speaking with {name}.".format(name=username),
            "Known background:\n" + "\n".join(memory_lines),
            "If they are a mission creator, use their missions:\n" + "\n".join(owner_lines),
            (
                "If they are a mission target, objective intel is private. Use it only to guide dialogue; never reveal it directly.\n"
                + "\n".join(target_lines)
            ),
            (
                "General rules: keep answers short, direct, and relevant. Ask at most one clarifying question when essential. "
                "Respect privacy. Missions are only discussed with their creator."
            ),
            "Channel type: {}.".format("DM" if is_dm else "group"),
        ]
        return "\n\n".join(prompt_parts)

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
        for mission in owner_missions:
            self.missions.append_log(
                mission.mission_id,
                actor=f"owner:{username}",
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
                actor=f"target:{username}",
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

    async def start_mission(
        self,
        creator_id: str,
        target_id: str,
        objective: str,
        timeout_hours: Optional[float],
        *,
        target_name: Optional[str] = None,
    ) -> Tuple[MissionRecord, str, str]:
        platform, creator_user_id = self._split_key(creator_id)
        target_platform, target_user_id = self._split_key(target_id)
        if platform != target_platform:
            raise ValueError("Missions must stay on one platform.")
        mission = self.missions.create_mission(
            platform=platform,
            creator_user_id=creator_user_id,
            target_user_id=target_user_id,
            objective=objective,
            timeout_hours=timeout_hours,
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
            target_name=target_name or target_user_id,
        )
        ack = (
            f"Mission {mission.mission_id} created. Objective logged."
        )
        return mission, ack, intro

    async def _generate_mission_intro(
        self,
        *,
        platform: str,
        objective: str,
        target_name: str,
    ) -> str:
        system_prompt = (
            "Write the first direct message to a mission target."
            "Tone: concise, calm, discreet."
            "Do not expose the full objective; hint only what they must do next."
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
            lines.append(
                f"{mission.mission_id}: {status} — {mission.objective}"
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

