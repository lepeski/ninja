import json
import logging
import math
import re
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI

log = logging.getLogger(__name__)


STORE_MIN_LEN = 18
STORE_COOLDOWN_S = 5 * 60
RECALL_TOPK = 5
RECALL_MAX_CHARS = 600
EMBED_TRUNCATE = 800
MEM_SUMMARY_COOLDOWN_S = 6 * 3600
MAX_HISTORY = 8
DM_MIN_INTERVAL = 0.0

MEM_FACT_PATTERNS_POS = [
    r"\bi like ([^.,;]+)",
    r"\bi love ([^.,;]+)",
    r"\bi enjoy ([^.,;]+)",
    r"\bmy favorite (?:game|food|thing|song|movie|band|color|sport) is ([^.,;]+)",
]
MEM_FACT_PATTERNS_NEG = [r"\bi (?:hate|dislike) ([^.,;]+)"]

MEM_Q_PATTERNS = {
    "whoami": {"any": ["who am i"]},
    "what_like": {
        "any": [
            "what do i like",
            "what do you know about me",
            "what do you remember about me",
            "tell me about myself",
        ]
    },
}

UNKNOWN_SUFFIX = "nigga"


@dataclass
class Agenda:
    goal: str
    steps: List[str]
    idx: int
    active: bool
    owner_id: str
    last_dm: float
    created: float
    warned: bool


class Memory:
    def __init__(self, db_path: str = "memory.db", mem_dir: Path = Path("mem")):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.short: Dict[Tuple[str, str], Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY)
        )
        self.mem_dir = mem_dir
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_dir = self.mem_dir / "inbox"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._last_store: Dict[Tuple[str, str], float] = {}
        self._last_profile: Dict[Tuple[str, str], float] = {}
        self._last_text: Dict[Tuple[str, str], str] = {}

    def _init_db(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agendas(
                  platform TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  goal TEXT NOT NULL,
                  steps TEXT NOT NULL,
                  idx INTEGER NOT NULL DEFAULT 0,
                  active INTEGER NOT NULL DEFAULT 1,
                  last_dm REAL NOT NULL DEFAULT 0,
                  owner_id TEXT NOT NULL DEFAULT '',
                  created REAL NOT NULL DEFAULT 0,
                  warned INTEGER NOT NULL DEFAULT 0,
                  PRIMARY KEY(platform, user_id)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users(
                  platform TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  alias TEXT DEFAULT '',
                  profile TEXT DEFAULT '',
                  profile_updated REAL DEFAULT 0,
                  PRIMARY KEY(platform, user_id)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  platform TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  ts REAL NOT NULL,
                  kind TEXT NOT NULL,
                  text TEXT NOT NULL,
                  embedding TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(platform, user_id)"
            )

    def _json_path(self, platform: str, user_id: str) -> Path:
        safe_platform = re.sub(r"[^a-z0-9_-]", "_", platform.lower())
        safe_user = re.sub(r"[^a-z0-9_-]", "_", str(user_id))
        return self.mem_dir / f"{safe_platform}_{safe_user}.json"

    def _load_json(self, platform: str, user_id: str) -> dict:
        path = self._json_path(platform, user_id)
        if not path.exists():
            data = {
                "platform": platform,
                "user_id": user_id,
                "alias": "",
                "facts": [],
                "notes": [],
                "history": [],
            }
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            return data
        try:
            return json.loads(path.read_text() or "{}")
        except Exception:
            data = {
                "platform": platform,
                "user_id": user_id,
                "alias": "",
                "facts": [],
                "notes": [],
                "history": [],
            }
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            return data

    def _save_json(self, platform: str, user_id: str, data: dict):
        path = self._json_path(platform, user_id)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def add_short(self, platform: str, channel_id: str, role: str, content: str):
        key = (platform, channel_id)
        self.short[key].append((role, content))

    def get_short(self, platform: str, channel_id: str) -> List[Tuple[str, str]]:
        key = (platform, channel_id)
        return list(self.short[key])

    def set_alias(self, platform: str, user_id: str, alias: str):
        alias = " ".join(alias.strip().split())[:64]
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO users(platform, user_id, alias, profile, profile_updated)
                VALUES(?,?,?,?,?)
                ON CONFLICT(platform, user_id)
                DO UPDATE SET alias=excluded.alias
                """,
                (platform, user_id, alias, "", 0.0),
            )
        data = self._load_json(platform, user_id)
        data["alias"] = alias
        self._save_json(platform, user_id, data)

    def get_alias(self, platform: str, user_id: str) -> str:
        cur = self.conn.execute(
            "SELECT alias FROM users WHERE platform=? AND user_id=?",
            (platform, user_id),
        )
        row = cur.fetchone()
        return row["alias"] if row and row["alias"] else ""

    def append_fact(self, platform: str, user_id: str, fact: str):
        fact = fact.strip()
        if not fact:
            return
        data = self._load_json(platform, user_id)
        if fact not in data["facts"]:
            data["facts"].append(fact)
            data["facts"] = data["facts"][-50:]
            self._save_json(platform, user_id, data)

    def append_history(self, platform: str, user_id: str, role: str, content: str):
        data = self._load_json(platform, user_id)
        data.setdefault("history", [])
        data["history"].append({"role": role, "content": content})
        data["history"] = data["history"][-200:]
        self._save_json(platform, user_id, data)

    def get_profile(self, platform: str, user_id: str) -> dict:
        data = self._load_json(platform, user_id)
        cur = self.conn.execute(
            "SELECT profile, profile_updated FROM users WHERE platform=? AND user_id=?",
            (platform, user_id),
        )
        row = cur.fetchone()
        profile = row["profile"] if row else ""
        updated = row["profile_updated"] if row else 0.0
        return {
            "alias": data.get("alias", ""),
            "facts": data.get("facts", []),
            "history": data.get("history", [])[-20:],
            "profile": profile,
            "profile_updated": updated,
        }

    def set_profile(self, platform: str, user_id: str, profile: str):
        now = time.time()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO users(platform, user_id, alias, profile, profile_updated)
                VALUES(?,?,?,?,?)
                ON CONFLICT(platform, user_id)
                DO UPDATE SET profile=excluded.profile, profile_updated=excluded.profile_updated
                """,
                (platform, user_id, "", profile, now),
            )

    def store_memory(
        self,
        platform: str,
        user_id: str,
        kind: str,
        text: str,
        embedding: Sequence[float],
    ):
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO memories(platform, user_id, ts, kind, text, embedding)
                VALUES(?,?,?,?,?,?)
                """,
                (platform, user_id, time.time(), kind, text, json.dumps(list(embedding))),
            )

    def recall(self, platform: str, user_id: str, embedding: Sequence[float], top_k: int) -> List[str]:
        cur = self.conn.execute(
            "SELECT text, embedding FROM memories WHERE platform=? AND user_id=? ORDER BY ts DESC LIMIT 200",
            (platform, user_id),
        )
        rows = cur.fetchall()
        scored: List[Tuple[float, str]] = []
        for row in rows:
            try:
                vec = json.loads(row["embedding"])
            except Exception:
                continue
            if not vec:
                continue
            sim = self._cosine(embedding, vec)
            if sim <= 0:
                continue
            scored.append((sim, row["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        texts: List[str] = []
        chars = 0
        for sim, text in scored[:top_k]:
            if chars + len(text) > RECALL_MAX_CHARS:
                break
            texts.append(text)
            chars += len(text)
        return texts

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        sa = math.sqrt(sum(x * x for x in a))
        sb = math.sqrt(sum(x * x for x in b))
        if sa == 0 or sb == 0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (sa * sb)

    def get_last_store(self, platform: str, user_id: str) -> float:
        return self._last_store.get((platform, user_id), 0.0)

    def set_last_store(self, platform: str, user_id: str):
        self._last_store[(platform, user_id)] = time.time()

    def get_last_profile(self, platform: str, user_id: str) -> float:
        return self._last_profile.get((platform, user_id), 0.0)

    def set_last_profile(self, platform: str, user_id: str):
        self._last_profile[(platform, user_id)] = time.time()

    def get_last_text(self, platform: str, user_id: str) -> str:
        return self._last_text.get((platform, user_id), "")

    def set_last_text(self, platform: str, user_id: str, text: str):
        self._last_text[(platform, user_id)] = text

    def has_assistant_reply(self, platform: str, user_id: str) -> bool:
        data = self._load_json(platform, user_id)
        return any(item.get("role") == "assistant" for item in data.get("history", []))

    def log_unsolicited_dm(self, platform: str, user_id: str, text: str):
        safe_platform = re.sub(r"[^a-z0-9_-]", "_", platform.lower())
        safe_user = re.sub(r"[^a-z0-9_-]", "_", str(user_id))
        path = self.inbox_dir / f"{safe_platform}_{safe_user}.log"
        entry = {
            "ts": time.time(),
            "platform": platform,
            "user_id": user_id,
            "text": text,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def set_agenda(
        self,
        platform: str,
        user_id: str,
        goal: str,
        steps: Sequence[str],
        owner_id: str,
    ):
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO agendas(platform, user_id, goal, steps, idx, active, last_dm, owner_id, created, warned)
                VALUES(?,?,?,?,0,1,0,?, ?, 0)
                ON CONFLICT(platform, user_id)
                DO UPDATE SET goal=excluded.goal, steps=excluded.steps, idx=0, active=1, last_dm=0, owner_id=excluded.owner_id, created=excluded.created, warned=0
                """,
                (
                    platform,
                    user_id,
                    goal,
                    json.dumps(list(steps)),
                    owner_id,
                    time.time(),
                ),
            )

    def clear_agenda(self, platform: str, user_id: str):
        with self.conn:
            self.conn.execute(
                "DELETE FROM agendas WHERE platform=? AND user_id=?",
                (platform, user_id),
            )

    def get_agenda(self, platform: str, user_id: str) -> Optional[Agenda]:
        cur = self.conn.execute(
            "SELECT goal, steps, idx, active, owner_id, last_dm, created, warned FROM agendas WHERE platform=? AND user_id=?",
            (platform, user_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        steps = []
        try:
            steps = json.loads(row["steps"]) or []
        except Exception:
            steps = []
        return Agenda(
            goal=row["goal"],
            steps=list(steps),
            idx=int(row["idx"] or 0),
            active=bool(row["active"]),
            owner_id=row["owner_id"],
            last_dm=float(row["last_dm"] or 0.0),
            created=float(row["created"] or 0.0),
            warned=bool(row["warned"]),
        )

    def update_agenda_progress(
        self, platform: str, user_id: str, idx: int, warned: bool = False
    ):
        with self.conn:
            self.conn.execute(
                "UPDATE agendas SET idx=?, warned=?, last_dm=? WHERE platform=? AND user_id=?",
                (idx, int(warned), time.time(), platform, user_id),
            )


class Assistant:
    def __init__(
        self,
        *,
        openai_api_key: str,
        model: str,
        embedding_model: str,
        mem_dir: str = "mem",
        memory_db: str = "memory.db",
    ):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.memory = Memory(db_path=memory_db, mem_dir=Path(mem_dir))
    async def close(self):
        pass

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _should_reply(self, text: str, is_dm: bool) -> Tuple[bool, str]:
        if is_dm:
            return True, text.strip()
        norm = text.strip()
        if not norm:
            return False, text
        lowered = norm.lower()
        if lowered.startswith("ninja"):
            cleaned = re.sub(r"^ninja[,:\s]*", "", norm, flags=re.IGNORECASE)
            return True, cleaned.strip()
        return False, text

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
        should_reply, content = self._should_reply(message, is_dm)
        if not should_reply:
            return None
        platform = platform.lower()
        user_id = str(user_id)
        channel_id = str(channel_id)
        norm_content = content.strip()
        if not norm_content:
            return None

        if is_dm and not self.memory.has_assistant_reply(platform, user_id):
            self.memory.log_unsolicited_dm(platform, user_id, norm_content)

        known_alias = self.memory.get_alias(platform, user_id)
        self.memory.set_alias(platform, user_id, username)
        self.memory.add_short(platform, channel_id, "user", norm_content)
        self.memory.append_history(platform, user_id, "user", norm_content)

        agenda = self.memory.get_agenda(platform, user_id) if is_dm else None

        special = self._detect_special(norm_content)
        if special:
            return await self._handle_special(
                special,
                platform=platform,
                user_id=user_id,
                username=username,
                agenda=agenda,
            )

        embed = await self._embed_text(norm_content)
        recalls = []
        if embed:
            recalls = self.memory.recall(platform, user_id, embed, RECALL_TOPK)
            await self._maybe_store_memory(
                platform, user_id, kind="observation", text=norm_content, embedding=embed
            )
            self._maybe_extract_facts(platform, user_id, norm_content, embed)

        prompt = self._build_prompt(
            platform=platform,
            user_id=user_id,
            username=username,
            message=norm_content,
            recalls=recalls,
            agenda=agenda if is_dm else None,
            is_dm=is_dm,
        )

        short_history = self.memory.get_short(platform, channel_id)
        messages = self._format_messages(prompt, short_history, norm_content)
        try:
            response = await self._chat(messages)
        except Exception as exc:
            log.exception("chat failure: %s", exc)
            return "I can't respond right now." + (
                f" {UNKNOWN_SUFFIX}" if not self.memory.get_alias(platform, user_id) else ""
            )

        self.memory.add_short(platform, channel_id, "assistant", response)
        self.memory.append_history(platform, user_id, "assistant", response)

        if agenda and agenda.active:
            idx = min(agenda.idx + 1, len(agenda.steps))
            self.memory.update_agenda_progress(platform, user_id, idx)

        if not known_alias:
            response = f"{response} {UNKNOWN_SUFFIX}".strip()

        return response

    def _detect_special(self, text: str) -> Optional[str]:
        norm = self._normalize(text)
        for key, patt in MEM_Q_PATTERNS.items():
            if any(trigger in norm for trigger in patt["any"]):
                return key
        return None

    async def _handle_special(
        self,
        special: str,
        *,
        platform: str,
        user_id: str,
        username: str,
        agenda: Optional[Agenda],
    ) -> str:
        profile = self.memory.get_profile(platform, user_id)
        alias = profile.get("alias") or username
        facts = profile.get("facts", [])
        if special == "whoami":
            if facts:
                return f"You are {alias}. I remember {', '.join(facts)}."
            return f"You are {alias}. I don't have more notes yet."
        if special == "what_like":
            if facts:
                return f"I have recorded that you value {', '.join(facts)}."
            return "I don't have any preferences saved yet."
        return "I don't have that information."

    def _maybe_extract_facts(
        self,
        platform: str,
        user_id: str,
        text: str,
        embedding: Optional[Sequence[float]] = None,
    ):
        lowered = text.lower()
        for pattern in MEM_FACT_PATTERNS_POS:
            match = re.search(pattern, lowered)
            if match:
                fact = match.group(1).strip()
                self.memory.append_fact(platform, user_id, f"likes {fact}")
                if embedding:
                    self.memory.store_memory(
                        platform, user_id, "fact", f"likes {fact}", embedding
                    )
        for pattern in MEM_FACT_PATTERNS_NEG:
            match = re.search(pattern, lowered)
            if match:
                fact = match.group(1).strip()
                self.memory.append_fact(platform, user_id, f"dislikes {fact}")
                if embedding:
                    self.memory.store_memory(
                        platform, user_id, "fact", f"dislikes {fact}", embedding
                    )

    async def _maybe_store_memory(
        self,
        platform: str,
        user_id: str,
        *,
        kind: str,
        text: str,
        embedding: Optional[Sequence[float]],
    ):
        if not embedding or len(text) < STORE_MIN_LEN:
            return
        last = self.memory.get_last_store(platform, user_id)
        if time.time() - last < STORE_COOLDOWN_S:
            return
        if text == self.memory.get_last_text(platform, user_id):
            return
        self.memory.store_memory(platform, user_id, kind, text, embedding)
        self.memory.set_last_store(platform, user_id)
        self.memory.set_last_text(platform, user_id, text)

    def _build_prompt(
        self,
        *,
        platform: str,
        user_id: str,
        username: str,
        message: str,
        recalls: List[str],
        agenda: Optional[Agenda],
        is_dm: bool,
    ) -> str:
        profile = self.memory.get_profile(platform, user_id)
        alias = profile.get("alias") or username
        facts = profile.get("facts", [])
        summary_lines = []
        if facts:
            summary_lines.append("Facts: " + "; ".join(facts))
        if recalls:
            summary_lines.append("Memories: " + " | ".join(recalls))
        if agenda and agenda.active:
            current_step = (
                agenda.steps[agenda.idx]
                if agenda.steps and agenda.idx < len(agenda.steps)
                else agenda.steps[-1] if agenda.steps else ""
            )
            summary_lines.append(
                f"Agenda goal: {agenda.goal}\nCurrent step: {current_step}"
            )
        summary = "\n".join(summary_lines).strip()

        persona = (
            "You are Ninja, a concise strategist."
            " Speak in brief, direct sentences with a steady tone."
            " Avoid metaphors and flowery language."
            " Keep responses focused on the user's needs."
        )
        prompt = (
            f"Persona: {persona}\n"
            f"Platform: {platform}\n"
            f"User alias: {alias}\n"
            f"Is DM: {is_dm}\n"
        )
        if summary:
            prompt += f"Known data:\n{summary}\n"
        prompt += "Respond briefly and clearly."
        return prompt

    def _format_messages(
        self,
        prompt: str,
        short_history: List[Tuple[str, str]],
        current: str,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": prompt},
        ]
        for role, content in short_history[-MAX_HISTORY:]:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": current})
        return messages

    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=350,
        )
        return completion.choices[0].message.content.strip()

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        text = text.strip()
        if not text:
            return None
        if len(text) > EMBED_TRUNCATE:
            text = text[:EMBED_TRUNCATE]
        try:
            result = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
        except Exception as exc:
            log.warning("embedding failure: %s", exc)
            return None
        return list(result.data[0].embedding)

    async def assign_agenda(
        self,
        *,
        platform: str,
        target_user_id: str,
        target_username: str,
        goal: str,
        owner_id: str,
    ) -> Tuple[str, str]:
        platform = platform.lower()
        target_user_id = str(target_user_id)
        owner_id = str(owner_id)
        steps = await self._generate_agenda_steps(goal)
        if not steps:
            steps = ["Take one specific action toward the goal today."]
        self.memory.set_agenda(platform, target_user_id, goal, steps, owner_id)
        step_lines = [f"{idx + 1}. {step}" for idx, step in enumerate(steps)]
        dm_message = "\n".join(
            [
                f"Mission assigned: {goal}",
                "",
                *step_lines,
                "",
                "Check in here when you make progress.",
            ]
        ).strip()
        ack = f"Mission set for {target_username}."
        return ack, dm_message

    async def stop_agenda(
        self,
        *,
        platform: str,
        target_user_id: str,
    ) -> str:
        platform = platform.lower()
        target_user_id = str(target_user_id)
        self.memory.clear_agenda(platform, target_user_id)
        return "Agenda cleared."

    async def _generate_agenda_steps(self, goal: str) -> List[str]:
        prompt = (
            "Create 3 concise steps for a personal mission."
            " Keep each under 120 characters."
        )
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": goal},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            text = completion.choices[0].message.content.strip()
        except Exception as exc:
            log.warning("agenda generation failed: %s", exc)
            return []
        steps = []
        for line in text.splitlines():
            clean = line.strip(" -*1234567890.\t")
            if clean:
                steps.append(clean.strip())
        return steps[:5]


__all__ = ["Assistant"]
