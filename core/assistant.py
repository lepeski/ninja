import json
import logging
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

MAX_HISTORY = 50
HISTORY_MAX_CHARS = 6000


@dataclass
class Mission:
    mission_id: str
    goal: str
    owner_id: str
    assigned_at: float
    status: str = "active"
    notes: Optional[str] = None


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

    def _default_profile(self, platform: str, user_id: str) -> dict:
        return {
            "platform": platform,
            "user_id": user_id,
            "alias": "",
            "preferences": {},
            "facts": {},
            "personality": [],
            "missions": [],
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
        profile.setdefault("missions", [])
        profile.setdefault("preferences", {})
        profile.setdefault("facts", {})
        profile.setdefault("personality", [])
        profile.setdefault("notes", [])
        profile["last_seen"] = time.time()
        self._save(platform, user_id, profile)
        return profile

    def _save(self, platform: str, user_id: str, data: dict) -> None:
        data = dict(data)
        data.setdefault("missions", [])
        data.setdefault("preferences", {})
        data.setdefault("facts", {})
        data.setdefault("personality", [])
        data.setdefault("notes", [])
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO user_profiles(platform, user_id, data)
                VALUES(?,?,?)
                ON CONFLICT(platform, user_id)
                DO UPDATE SET data=excluded.data
                """,
                (platform, user_id, json.dumps(data, ensure_ascii=False)),
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
            if value not in profile.setdefault("personality", []):
                profile["personality"].append(value)
        else:
            notes = profile.setdefault("notes", [])
            entry = {"key": key, "value": value, "ts": time.time()}
            notes.append(entry)
            profile["notes"] = notes[-50:]
        self._save(platform, user_id, profile)

    def add_mission(
        self, platform: str, user_id: str, mission: Mission
    ) -> Mission:
        profile = self.recall(platform, user_id)
        missions = profile.setdefault("missions", [])
        missions = [m for m in missions if m.get("status", "active") == "active"]
        missions.append(
            {
                "mission_id": mission.mission_id,
                "goal": mission.goal,
                "owner_id": mission.owner_id,
                "assigned_at": mission.assigned_at,
                "status": mission.status,
                "notes": mission.notes or "",
            }
        )
        profile["missions"] = missions
        self._save(platform, user_id, profile)
        return mission

    def update_mission_status(
        self, platform: str, user_id: str, status: str
    ) -> int:
        profile = self.recall(platform, user_id)
        missions = profile.setdefault("missions", [])
        count = 0
        for mission in missions:
            if mission.get("status", "active") == "active":
                mission["status"] = status
                mission["updated_at"] = time.time()
                count += 1
        profile["missions"] = missions
        self._save(platform, user_id, profile)
        return count

    def log_history(self, platform: str, channel_id: str, role: str, content: str) -> None:
        key = (platform, channel_id)
        history = self._history[key]
        history.append((role, content))
        total_chars = sum(len(item[1]) for item in history)
        while total_chars > HISTORY_MAX_CHARS and len(history) > 1:
            popped = history.popleft()
            total_chars -= len(popped[1])

    def get_history(self, platform: str, channel_id: str) -> List[Tuple[str, str]]:
        return list(self._history[(platform, channel_id)])

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
                f"platform: {platform}\nuser_id: {user_id}\nusername: {username}\nts: {ts}\nmessage: {message}\n",
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Failed to archive first contact DM: %s", exc)


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
        self.persona_prompt = self._load_persona_prompt()

    def _load_persona_prompt(self) -> str:
        return (
            "You are Ninja, a multi-platform conversational agent who speaks with human warmth, wit, and brevity. "
            "All behavioral guidance lives in this prompt. Blend confidence with empathy, adapt to the user's tone, and stay naturally conversational. "
            "Remember relevant details about people, weave missions into organic dialogue, and never default to rigid scripts or repeated catchphrases. "
            "Group chats only receive replies when a message explicitly starts with your trigger word 'ninja'. "
            "In every exchange you should: maintain context, draw on stored memories, pursue active missions with subtlety, ask natural clarifying questions when needed, and allow humor or curiosity."
        )

    async def close(self) -> None:
        await self.client.close()

    async def handle_message(
        self,
        *,
        platform: str,
        user_id: str,
        username: str,
        channel_id: str,
        message: str,
        is_dm: bool,
    ) -> Optional[str]:
        if not message:
            return None
        trimmed = message.strip()
        if not is_dm:
            lowered = trimmed.lower()
            if not lowered.startswith("ninja"):
                return None
            trimmed = trimmed[len("ninja") :].lstrip(" ,:;\n")
            if not trimmed:
                return None
        else:
            self.memory.record_first_contact(platform, user_id, username, trimmed)
        reply = await self.send(
            platform=platform,
            user_id=user_id,
            username=username,
            channel_id=channel_id,
            message=trimmed,
            is_dm=is_dm,
        )
        return reply

    async def send(
        self,
        *,
        platform: str,
        user_id: str,
        username: str,
        channel_id: str,
        message: str,
        is_dm: bool,
    ) -> Optional[str]:
        profile = self.memory.recall(platform, user_id)
        missions = [
            m for m in profile.get("missions", []) if m.get("status", "active") == "active"
        ]
        system_prompt = self._build_system_prompt(username, profile, missions)
        history = self.memory.get_history(platform, channel_id)
        messages = [{"role": "system", "content": system_prompt}]
        for role, content in history:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                top_p=0.9,
            )
        except Exception as exc:
            log.exception("OpenAI chat failure: %s", exc)
            return "I'm having trouble thinking right now."
        reply = response.choices[0].message.content or ""
        self.memory.log_history(platform, channel_id, "user", message)
        self.memory.log_history(platform, channel_id, "assistant", reply)
        await self._extract_memories(
            platform=platform,
            user_id=user_id,
            username=username,
            last_user=message,
            last_reply=reply,
            profile=profile,
        )
        return reply

    def _build_system_prompt(
        self, username: str, profile: dict, missions: List[dict]
    ) -> str:
        memory_lines: List[str] = []
        prefs = profile.get("preferences", {})
        if prefs:
            for key, value in prefs.items():
                memory_lines.append(f"Preference - {key}: {value}")
        facts = profile.get("facts", {})
        if facts:
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
        mission_lines: List[str] = []
        for mission in missions:
            goal = mission.get("goal", "")
            mission_lines.append(f"Active mission goal: {goal}")
        if not mission_lines:
            mission_lines.append("No active missions.")
        memory_block = "\n".join(memory_lines)
        mission_block = "\n".join(mission_lines)
        return (
            f"{self.persona_prompt}\n\n"
            f"You are currently speaking with {username}.\n"
            f"Known background about them:\n{memory_block}\n\n"
            f"Mission context for this relationship:\n{mission_block}\n\n"
            "Respond like a thoughtful friend who remembers the past. Keep replies concise but expressive. "
            "If you need more intel for a mission, naturally ask the assigning owner when they speak to you, otherwise guide the conversation with the user you are talking to."
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
            "You review the latest user message and assistant reply to decide if anything should be saved as long-term memory. "
            "Return a JSON array of memory items to store. Each item should be an object with keys: 'category' (preferences|facts|personality|notes), 'key', and 'value'. "
            "If nothing is worth storing, return an empty JSON array."
        )
        prompt = [
            {"role": "system", "content": extractor_system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "platform": platform,
                        "user_id": user_id,
                        "username": username,
                        "user_message": last_user,
                        "assistant_reply": last_reply,
                        "existing_memory": profile,
                    }
                ),
            },
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

    async def assign_agenda(
        self,
        *,
        platform: str,
        target_user_id: str,
        target_username: str,
        goal: str,
        owner_id: str,
    ) -> Tuple[str, str]:
        mission = Mission(
            mission_id=str(int(time.time() * 1000)),
            goal=goal.strip(),
            owner_id=owner_id,
            assigned_at=time.time(),
        )
        self.memory.add_mission(platform, target_user_id, mission)
        dm_text = await self._generate_mission_intro(
            platform=platform,
            target_username=target_username,
            goal=goal,
        )
        ack = f"Mission logged for {target_username}. I'll approach them discreetly."
        return ack, dm_text

    async def _generate_mission_intro(
        self,
        *,
        platform: str,
        target_username: str,
        goal: str,
    ) -> str:
        system_prompt = (
            "You craft the very first direct message that Ninja sends to a mission target. "
            "Be brief, calm, and hint at the mission without revealing sensitive details. "
            "Invite collaboration and sound human."
        )
        prompt = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target": target_username,
                        "mission_goal": goal,
                        "platform": platform,
                    }
                ),
            },
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0.7,
            )
        except Exception as exc:
            log.exception("Mission intro generation failed: %s", exc)
            return "I have something important for us to tackle. Are you up for a quick mission?"
        return response.choices[0].message.content or "I have something important for us to tackle. Are you up for a quick mission?"

    async def stop_agenda(
        self,
        *,
        platform: str,
        target_user_id: str,
    ) -> str:
        count = self.memory.update_mission_status(platform, target_user_id, "stopped")
        if count:
            return "Mission status updated. I'll ease off for now."
        return "No active missions were found for that user."

    def remember(
        self, *, platform: str, user_id: str, key: str, value: str, category: str = "notes"
    ) -> None:
        self.memory.remember(platform, user_id, key, value, category=category)

    def recall(self, *, platform: str, user_id: str) -> dict:
        return self.memory.recall(platform, user_id)
