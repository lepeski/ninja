#!/usr/bin/env python3
# ninja discord bot — calm samurai agent (final + hybrid JSON memory)
# features: multi-user aware, presence Q&A, identity fixes, level-3 memory (embeddings),
# concise T3 tone, M1 memory answers (+F3 fallback), S3 hybrid memory, passive opinions (P2,R1),
# DM missions, suffix-for-unknown users ("not"), strict address triggers ("ninja" summons),
# H2 hybrid memory (regex + LLM) and per-user human-readable JSON memories with history (U2)

from __future__ import annotations
import asyncio, contextlib, json, logging, math, os, re, random, sqlite3, time, pathlib
from collections import defaultdict, deque
from typing import Dict, Deque, List, Optional, Tuple

import aiohttp, discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ----- config -----
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("ninja")

DISCORD_TOKEN   = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL           = os.getenv("MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

OWNER_IDS       = {int(x) for x in os.getenv("OWNER_IDS", "").split(",") if x.strip().isdigit()}
ALLOWED_CHANNELS= {x.strip() for x in os.getenv("ALLOW_CHANNELS", "").split(",") if x.strip()}
GUILD_ID        = int(os.getenv("GUILD_ID", "465454837400731648"))

if not DISCORD_TOKEN or not OPENAI_API_KEY:
    raise SystemExit("missing DISCORD_TOKEN or OPENAI_API_KEY")

# pacing / limits
USER_RATE_SECONDS = 1
CHANNEL_RATE_SECONDS = 5
CHANNEL_RATE_LIMIT = 10
DM_MIN_INTERVAL = 0.0
MAX_HISTORY = 8
SILENCE_DURATION = 600  # 10m

# memory knobs
STORE_MIN_LEN = 6
STORE_COOLDOWN_S = 60
RECALL_TOPK = 5
RECALL_MAX_CHARS = 600
EMBED_TRUNCATE = 800
MEM_SUMMARY_COOLDOWN_S = 6*3600

# passive opinions
PASSIVE_CHANCE = float(os.getenv("PASSIVE_TRIGGER_CHANCE", "0.05"))  # P2 ~5%

# suffix for unknown users (acronym of "Nigel Inca Gang Gang Adam" -> "nigga")
UNKNOWN_SUFFIX = "nigga"  # lowercase, no periods, no spaces

# json memory dir
MEM_DIR = pathlib.Path(os.getenv("MEM_DIR", "mem"))
MEM_DIR.mkdir(parents=True, exist_ok=True)

# memory extraction patterns (facts) — regex (reliable)
MEM_FACT_PATTERNS_POS = [
    r"\bi like ([^.,;]+)", r"\bi love ([^.,;]+)", r"\bi enjoy ([^.,;]+)",
    r"\bmy favorite (?:game|food|thing|song|movie|band|color|sport) is ([^.,;]+)"
]
MEM_FACT_PATTERNS_NEG = [r"\bi (?:hate|dislike) ([^.,;]+)"]

# memory-question triggers — intercept (deterministic)
MEM_Q_PATTERNS = {
    "whoami": {"any": ["who am i"]},
    "what_like": {"any": ["what do i like","what do you know about me","what do you remember about me","tell me about myself"]},
}

# ----- utils -----
def _now() -> float: return time.time()
def _clamp_text(s: str, n: int) -> str: return s if len(s) <= n else s[:n]
def _norm(s:str) -> str: return re.sub(r"\s+", " ", (s or "").strip().lower())

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b): return 0.0
    sa = math.sqrt(sum(x*x for x in a)); sb = math.sqrt(sum(x*x for x in b))
    if sa == 0 or sb == 0: return 0.0
    return sum(x*y for x,y in zip(a,b)) / (sa*sb)

# ----- memory (sqlite + human-readable JSON per user) -----
class Memory:
    def __init__(self, path="memory.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.short: Dict[Tuple[int,int], Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
        self._init_db()
        self._last_store: Dict[int, float] = {}
        self._last_profile: Dict[int, float] = {}

    # ----- sqlite init -----
    def _init_db(self):
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agendas(
              user_id INTEGER PRIMARY KEY,
              agenda TEXT NOT NULL,
              steps TEXT NOT NULL,
              idx INTEGER NOT NULL DEFAULT 0,
              active INTEGER NOT NULL DEFAULT 1,
              last_dm REAL NOT NULL DEFAULT 0,
              owner_id INTEGER NOT NULL DEFAULT 0,
              created REAL NOT NULL DEFAULT 0,
              warned INTEGER NOT NULL DEFAULT 0
            )""")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users(
              user_id INTEGER PRIMARY KEY,
              alias TEXT DEFAULT '',
              profile TEXT DEFAULT '',
              profile_updated REAL DEFAULT 0
            )""")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              ts REAL NOT NULL,
              kind TEXT NOT NULL,
              text TEXT NOT NULL,
              embedding TEXT NOT NULL
            )""")

    # ----- json paths & helpers -----
    def _json_path(self, uid:int) -> pathlib.Path:
        return MEM_DIR / f"{uid}.json"

    def _load_json(self, uid:int) -> dict:
        p = self._json_path(uid)
        if not p.exists():
            data = {"user_id": uid, "alias":"", "profile":"", "facts":[], "notes":[], "history":[]}
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            return data
        try:
            return json.loads(p.read_text() or "{}")
        except Exception:
            data = {"user_id": uid, "alias":"", "profile":"", "facts":[], "notes":[], "history":[]}
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            return data

    def _save_json(self, uid:int, data:dict):
        p = self._json_path(uid)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # ----- short memory (RAM) -----
    def add_short(self, chan_id:int, user_id:int, role:str, content:str):
        self.short[(chan_id,user_id)].append((role, content))
    def get_short(self, chan_id:int, user_id:int):
        return list(self.short[(chan_id,user_id)])

    # ----- users (alias/profile) with JSON mirror -----
    def set_alias(self, uid:int, alias:str):
        alias = " ".join(alias.strip().split()[:2])[:32]
        with self.conn:
            self.conn.execute(
                "INSERT INTO users(user_id,alias,profile,profile_updated) VALUES(?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET alias=excluded.alias",
                (uid, alias, "", 0)
            )
        data = self._load_json(uid)
        data["alias"] = alias
        self._save_json(uid, data)

    def get_alias(self, uid:int) -> Optional[str]:
        row = self.conn.execute("SELECT alias FROM users WHERE user_id=?", (uid,)).fetchone()
        if row and row["alias"]:
            return row["alias"]
        # fallback to JSON if DB empty
        data = self._load_json(uid)
        return data.get("alias") or None

    def get_alias_safe(self, uid:int) -> str:
        return self.get_alias(uid) or ""

    def set_profile(self, uid:int, profile:str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO users(user_id,alias,profile,profile_updated) VALUES(?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET profile=excluded.profile, profile_updated=excluded.profile_updated",
                (uid, self.get_alias(uid) or "", profile, _now())
            )
        data = self._load_json(uid)
        old = (data.get("profile") or "").strip()
        if old and _norm(old) != _norm(profile):
            data.setdefault("history", []).append(f"old_profile: {old}")
        data["profile"] = profile
        self._save_json(uid, data)

    def get_profile(self, uid:int) -> str:
        row = self.conn.execute("SELECT profile FROM users WHERE user_id=?", (uid,)).fetchone()
        if row and row["profile"]:
            return row["profile"]
        data = self._load_json(uid)
        return data.get("profile","")

    # ----- long-term memories (sqlite + JSON facts U2) -----
    def can_store(self, uid:int) -> bool:
        last = self._last_store.get(uid, 0)
        return (_now() - last) >= STORE_COOLDOWN_S
    def note_stored(self, uid:int): self._last_store[uid] = _now()
    def last_profile_time(self, uid:int) -> float:
        return self._last_profile.get(uid, 0)
    def note_profile_time(self, uid:int): self._last_profile[uid] = _now()

    def add_memory(self, uid:int, kind:str, text:str, embedding:List[float]):
        with self.conn:
            self.conn.execute(
                "INSERT INTO memories(user_id, ts, kind, text, embedding) VALUES(?,?,?,?,?)",
                (uid, _now(), kind, text, json.dumps(embedding))
            )
        # optional JSON note mirror for human review
        if kind in ("note","event"):
            data = self._load_json(uid)
            data.setdefault("notes", []).append(text)
            # cap notes to avoid infinite growth
            if len(data["notes"]) > 200:
                data["notes"] = data["notes"][-200:]
            self._save_json(uid, data)

    def _json_fact_upsert(self, uid:int, fact:str):
        """U2: keep history; replace opposing fact; deduplicate."""
        data = self._load_json(uid)
        facts = data.setdefault("facts", [])
        hist  = data.setdefault("history", [])
        nf = _norm(fact)
        # dedupe
        if any(_norm(f)==nf for f in facts): 
            self._save_json(uid, data); return
        # simple opposition check: "likes X" vs "dislikes X"
        m1 = re.match(r"(likes|dislikes)\s+(.+)", nf)
        if m1:
            pol, item = m1.group(1), m1.group(2).strip()
            opposite = "dislikes" if pol=="likes" else "likes"
            opp = f"{opposite} {item}"
            kept = []
            moved = False
            for f in facts:
                if _norm(f) == _norm(opp):
                    hist.append(f)  # move old to history
                    moved = True
                else:
                    kept.append(f)
            facts = kept
            if moved:
                data["facts"] = facts
        facts.append(fact)
        # cap
        if len(facts) > 200: facts = facts[-200:]
        data["facts"] = facts
        data["history"] = hist
        self._save_json(uid, data)

    def add_fact(self, uid:int, text:str, embedding:List[float]):
        self.add_memory(uid, "fact", text, embedding)
        self._json_fact_upsert(uid, text)

    def get_facts(self, uid:int, limit:int=24) -> List[str]:
        # prefer JSON for human-curated order; backfill with sqlite if JSON missing
        data = self._load_json(uid)
        facts = data.get("facts", [])
        if facts:
            return facts[-limit:]
        rows = self.conn.execute(
            "SELECT text FROM memories WHERE user_id=? AND kind='fact' ORDER BY ts DESC LIMIT ?",
            (uid, limit)
        ).fetchall()
        return [r["text"] for r in rows]

    def recall(self, uid:int, query_vec:List[float], topk:int=RECALL_TOPK) -> List[str]:
        rows = self.conn.execute(
            "SELECT text, embedding FROM memories WHERE user_id=? ORDER BY ts DESC LIMIT 400", (uid,)
        ).fetchall()
        scored: List[Tuple[float,str]] = []
        for r in rows:
            try:
                vec = json.loads(r["embedding"])
                scored.append((cosine(query_vec, vec), r["text"]))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for s,t in scored[:topk] if s > 0.2]

    # ----- agendas -----
    def get_agenda(self, uid:int):
        return self.conn.execute("SELECT * FROM agendas WHERE user_id=? AND active=1", (uid,)).fetchone()
    def upsert_agenda(self, target_id:int, owner_id:int, agenda:str, steps:List[str]):
        now = _now()
        with self.conn:
            self.conn.execute("""
            INSERT INTO agendas(user_id, agenda, steps, idx, active, last_dm, owner_id, created, warned)
            VALUES(?, ?, ?, 0, 1, 0, ?, ?, 0)
            ON CONFLICT(user_id) DO UPDATE SET
              agenda=excluded.agenda, steps=excluded.steps, idx=0, active=1, last_dm=0,
              owner_id=excluded.owner_id, created=excluded.created, warned=0
            """, (target_id, agenda, json.dumps(steps), owner_id, now))
    def stop_agenda(self, uid:int):
        with self.conn:
            self.conn.execute("UPDATE agendas SET active=0 WHERE user_id=?", (uid,))
    def advance(self, uid:int):
        with self.conn:
            self.conn.execute("UPDATE agendas SET idx=idx+1 WHERE user_id=?", (uid,))
    def touch(self, uid:int):
        with self.conn:
            self.conn.execute("UPDATE agendas SET last_dm=? WHERE user_id=?", (_now(), uid))
    def last_dm(self, uid:int) -> float:
        row = self.get_agenda(uid); return float(row["last_dm"] or 0) if row else 0.0
    def active_rows(self):
        return self.conn.execute("SELECT * FROM agendas WHERE active=1").fetchall()
    def set_warned(self, uid:int):
        with self.conn:
            self.conn.execute("UPDATE agendas SET warned=1 WHERE user_id=?", (uid,))

# ----- identity (public multi-user awareness) -----
class Identity:
    def __init__(self):
        self.alias: Dict[int, str] = {}
        self.seen_by_channel: Dict[int, set] = defaultdict(set)
    def note_message(self, chan_id:int, user:discord.User, text:str):
        self.seen_by_channel[chan_id].add(user.id)
        low = (text or "").strip().lower()
        name = None
        if low.startswith("i'm "): name = text.split(" ",1)[1].strip()
        elif low.startswith("i am "):
            parts = text.split(" ",2)
            name = parts[2].strip() if len(parts)>2 else ""
        elif low.startswith("my name is "): name = text.split("my name is ",1)[1].strip()
        if name:
            name = name.strip().strip(".!,?")[:32]
            toks = name.split()
            self.alias[user.id] = name if len(toks) <= 2 else toks[0]
    def who_am_i(self, user:discord.User, db_alias:Optional[str]) -> str:
        return self.alias.get(user.id) or db_alias or (user.display_name or user.name)
    def list_here(self, chan_id:int, lookup_alias) -> List[str]:
        out = []
        for uid in sorted(self.seen_by_channel.get(chan_id, set())):
            name = lookup_alias(uid) or self.alias.get(uid)
            out.append(name or str(uid))
        return out
    def is_here(self, chan_id:int, name:str, lookup_alias) -> bool:
        name = name.lower()
        for uid in self.seen_by_channel.get(chan_id, set()):
            if (lookup_alias(uid) or self.alias.get(uid,"")).lower() == name: return True
        return False

# ----- rate limiter -----
class RateLimiter:
    def __init__(self):
        self.user_ts: Dict[int, float] = {}
        self.chan_ts: Dict[int, Deque[float]] = defaultdict(deque)
    def allow(self, user_id:int, chan_id:int) -> bool:
        now = time.monotonic()
        t = self.user_ts.get(user_id)
        if t and now - t < USER_RATE_SECONDS: return False
        q = self.chan_ts[chan_id]
        while q and now - q[0] > CHANNEL_RATE_SECONDS: q.popleft()
        if len(q) >= CHANNEL_RATE_LIMIT: return False
        self.user_ts[user_id] = now; q.append(now); return True

# ----- bot -----
class Ninja(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default(); intents.message_content = True
        super().__init__(command_prefix="/", intents=intents)
        self.mem = Memory()
        self.rate = RateLimiter()
        self.oa = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.session = aiohttp.ClientSession()
        self.silenced_users: Dict[int, float] = {}
        self.idm = Identity()

    # tone helpers (T3)
    def opener(self) -> str:   return "my brothers sent me. we are a quiet order. ready?"
    def reassure(self) -> str: return "breathe. speak simply."
    def clarify(self) -> str:  return "one clear thought."
    def closing(self) -> str:  return "thx. adios ^-^/"
    def reveal(self) -> str:   return "we move unseen for balance. your part matters."

    # addressing: only respond if addressed; "ninja" always summons
    def _is_addressed(self, m: discord.Message) -> bool:
        if isinstance(m.channel, discord.DMChannel): return True
        if self.user in getattr(m, "mentions", []): return True
        if m.reference and getattr(m.reference, "resolved", None):
            try:
                return m.reference.resolved.author.id == self.user.id
            except Exception:
                pass
        t = (m.content or "").strip().lower()
        return t.startswith("ninja")  # "ninja", "ninja,", "ninja:"

    # suffix (Scope A: everywhere bot speaks), once per message
    def _needs_suffix(self, uid:int) -> bool:
        return not (self.mem.get_alias(uid) or self.idm.alias.get(uid))
    def _decorate_unknown_once(self, uid:int, text:str) -> str:
        if not text or not self._needs_suffix(uid): return text
        stripped = text.strip()
        core = re.sub(r'[.!?]+$', '', stripped)
        return f"{core} {UNKNOWN_SUFFIX}."

    async def send_lc(self, dest: discord.abc.Messageable, text: str):
        txt = (text or "").strip().lower()
        if not txt: return
        with contextlib.suppress(Exception): await dest.send(txt)

    # embeddings
    async def embed(self, text:str) -> List[float]:
        text = _clamp_text(text, EMBED_TRUNCATE)
        r = await self.oa.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return r.data[0].embedding  # type: ignore

    # public replies with recall (<=2 short sentences; no "how can i help")
    async def gen_public(self, uid:int, text: str, hist: List[Tuple[str,str]], recall_snips: List[str]) -> str:
        sys = (
            "calm samurai helper. short, poetic, decisive. under 2 short sentences total. "
            "no filler. no 'how can i help'. stay on topic. lowercase."
        )
        ctx = ""
        if recall_snips:
            joined = " | ".join(_clamp_text(s, 120) for s in recall_snips)
            ctx = f"\n\nif relevant, you must incorporate these: {joined}"
        msgs = [{"role":"system","content":sys + ctx}]
        for role, cont in hist[-6:]: msgs.append({"role": role, "content": cont})
        msgs.append({"role":"user","content": text})
        try:
            r = await self.oa.chat.completions.create(model=MODEL, messages=msgs, temperature=0.4, stream=False)
            out = (r.choices[0].message.content or "").strip().lower() or "ok."
            return out
        except Exception as e:
            log.warning("public gen failed: %s", e); return "sorry."

    # planners (concise, mission-locked)
    async def plan_steps(self, agenda: str, k_min:int=2, k_max:int=5) -> List[str]:
        sys = ("you are a calm samurai guide. craft mission questions, one per line. "
               "each <= 10 words. no fluff. lowercase. "
               f"produce {k_min}..{k_max} lines.")
        try:
            r = await self.oa.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":sys},{"role":"user","content":agenda}],
                temperature=0.5, stream=False)
            text = (r.choices[0].message.content or "").strip().lower()
            steps = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
            if not steps:
                steps = [f"what matters most about {agenda}?", "what is the next step?"]
            return steps[:k_max]
        except Exception as e:
            log.warning("planner failed: %s", e)
            return [f"what matters most about {agenda}?", "what is the next step?"]

    async def rephrase_question(self, prev_q:str) -> str:
        sys = "rephrase simpler. <= 8 words. lowercase. mission-first. no fluff."
        try:
            r = await self.oa.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":sys},{"role":"user","content":prev_q}],
                temperature=0.2, stream=False)
            return (r.choices[0].message.content or "").strip().lower()
        except Exception:
            return prev_q

    async def followup_question(self, agenda:str, prior_q:str, user_reply:str) -> str:
        sys = "ask one short mission follow-up. <= 10 words. lowercase."
        u = f"mission: {agenda}\nprevious question: {prior_q}\nanswer: {user_reply}\nnext question:"
        try:
            r = await self.oa.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":sys},{"role":"user","content":u}],
                temperature=0.4, stream=False)
            return (r.choices[0].message.content or "").strip().lower()
        except Exception:
            return "what follows?"

    async def compress_result(self, agenda:str, transcript:str) -> str:
        sys = "summarize final outcome as short noun phrase, <=6 words, lowercase."
        u = f"mission: {agenda}\nconversation:\n{transcript}"
        try:
            r = await self.oa.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":sys},{"role":"user","content":u}],
                temperature=0.2, stream=False)
            return (r.choices[0].message.content or "").strip().lower()
        except Exception:
            return "result noted"

    async def report_owner(self, owner_id:int, target:discord.User, result:str):
        try:
            owner = await self.fetch_user(owner_id)
            await self.send_lc(owner, f"mission complete — target: {target.display_name} — result: {result}")
        except Exception as e:
            log.warning(f"report_owner failed: {e}")

    # memory Q detection / direct answer (M1 + F3)
    def _is_mem_query(self, txt:str) -> Optional[str]:
        t = txt.strip().lower()
        if any(k in t for k in MEM_Q_PATTERNS["whoami"]["any"]): return "whoami"
        if any(k in t for k in MEM_Q_PATTERNS["what_like"]["any"]): return "what_like"
        return None
    def _answer_memory_direct(self, uid:int) -> str:
        alias = self.mem.get_alias_safe(uid)
        facts = self.mem.get_facts(uid)
        prof = (self.mem.get_profile(uid) or "").strip()
        if alias and (facts or prof):
            facts_str = "; ".join(facts[:4])
            if facts_str: return f"you are {alias}. {facts_str}."
            return f"you are {alias}."
        if facts: return f"{'; '.join(facts[:4])}."
        return "no record yet. speak one truth about yourself."

    # fact extraction (S3 hybrid regex)
    async def _extract_and_store_facts_regex(self, uid:int, text:str, *, conservative:bool):
        low = (text or "").lower()
        found: List[Tuple[str,str]] = []
        for pat in MEM_FACT_PATTERNS_POS:
            m = re.search(pat, low)
            if m:
                val = m.group(1).strip()
                if val: found.append(("pos", val))
        if not conservative:
            for pat in MEM_FACT_PATTERNS_NEG:
                m = re.search(pat, low)
                if m:
                    val = m.group(1).strip()
                    if val: found.append(("neg", val))
        if not found: return
        for kind, v in found:
            stmt = f"likes {v}" if kind=="pos" else f"dislikes {v}"
            vec = await self.embed(stmt)
            self.mem.add_fact(uid, stmt, vec)

    # LLM-assisted fact extraction (H2)
    async def _extract_and_store_facts_ai(self, uid:int, text:str):
        """Ask the model to extract 0..5 stable first-person facts, return as JSON list of strings."""
        low = (text or "").strip()
        if not low: return
        sys = (
            "extract up to 5 stable first-person facts about the speaker from the message. "
            "only include durable preferences, dislikes, favorites, goals, or traits. "
            "return a pure JSON array of short lowercase strings, no narration."
        )
        try:
            r = await self.oa.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":sys},{"role":"user","content":low}],
                temperature=0.1, stream=False)
            raw = (r.choices[0].message.content or "").strip()
            # try to parse array
            arr = []
            try:
                arr = json.loads(raw)
                if not isinstance(arr, list): arr = []
            except Exception:
                # fallback: split lines
                arr = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip()]
            for item in arr[:5]:
                s = _norm(str(item))
                if not s or len(s) < 3: continue
                # coerce to "likes X" / "dislikes X" / simple trait
                if s.startswith(("i like ","i love ","i enjoy ")):
                    v = s.split(" ", 2)[2] if " " in s[2:] else s
                    stmt = f"likes {v}".strip()
                elif s.startswith(("i hate ","i dislike ")):
                    v = s.split(" ", 2)[2] if " " in s[2:] else s
                    stmt = f"dislikes {v}".strip()
                else:
                    stmt = s
                vec = await self.embed(stmt)
                self.mem.add_fact(uid, stmt, vec)
        except Exception as e:
            log.warning(f"ai fact extraction failed: {e}")

    # passive opinions (P2,R1): short, relevant, samurai tone; only when NOT addressed
    async def _maybe_passive_opinion(self, m: discord.Message):
        if self._is_addressed(m): return
        if random.random() >= PASSIVE_CHANCE: return
        low = (m.content or "").strip().lower()[:200]
        sys = ("calm samurai observer. one short opinion. <=10 words. "
               "no questions. no commands. lowercase. relevant to the message.")
        try:
            r = await self.oa.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":sys},
                          {"role":"user","content": low or "offer a brief observation"}],
                temperature=0.6, stream=False)
            out = (r.choices[0].message.content or "").strip().lower()
            if not out: return
            out = self._decorate_unknown_once(m.author.id, out)
            await self.send_lc(m.channel, out)
        except Exception:
            return

    # ----- events / commands -----
    async def setup_hook(self):
        self.setup_cmds()
        guild = discord.Object(id=GUILD_ID)
        await self.tree.sync(guild=guild)
        log.info(f"slash commands synced to guild {GUILD_ID}")
        asyncio.create_task(self.agenda_watchdog())
        asyncio.create_task(self.profile_refresher())

    def setup_cmds(self):
        def _allowed(inter: discord.Interaction) -> bool:
            if not ALLOWED_CHANNELS: return True
            ch_id = getattr(inter.channel, "id", None)
            return ch_id is not None and str(ch_id) in ALLOWED_CHANNELS

        @self.tree.command(name="assignagenda", description="start a mission (owner only)")
        @app_commands.describe(user="target user", agenda="short goal text")
        async def assignagenda(inter: discord.Interaction, user: discord.User, agenda: str):
            if not _allowed(inter):
                msg = self._decorate_unknown_once(inter.user.id, "not allowed in this channel")
                await inter.response.send_message(msg, ephemeral=True); return
            if OWNER_IDS and inter.user.id not in OWNER_IDS:
                msg = self._decorate_unknown_once(inter.user.id, "owner only")
                await inter.response.send_message(msg, ephemeral=True); return
            await inter.response.defer(ephemeral=True)
            steps = await self.plan_steps(agenda, 2, 5)
            steps.insert(0, self.opener())
            self.mem.upsert_agenda(user.id, inter.user.id, agenda, steps)
            msg = self._decorate_unknown_once(user.id, steps[0])
            await self.send_lc(user, msg)
            ok = self._decorate_unknown_once(inter.user.id, f"quest begun for {user.display_name}")
            await inter.followup.send(ok, ephemeral=True)

        @self.tree.command(name="stopagenda", description="stop mission (owner only)")
        async def stopagenda(inter: discord.Interaction, user: discord.User):
            if not _allowed(inter):
                msg = self._decorate_unknown_once(inter.user.id, "not allowed in this channel")
                await inter.response.send_message(msg, ephemeral=True); return
            if OWNER_IDS and inter.user.id not in OWNER_IDS:
                msg = self._decorate_unknown_once(inter.user.id, "owner only")
                await inter.response.send_message(msg, ephemeral=True); return
            self.mem.stop_agenda(user.id)
            msg = self._decorate_unknown_once(inter.user.id, "stopped")
            await inter.response.send_message(msg, ephemeral=True)

        @self.tree.command(name="whoishere", description="list present users (public)")
        async def whoishere(inter: discord.Interaction):
            if not _allowed(inter):
                msg = self._decorate_unknown_once(inter.user.id, "not allowed in this channel")
                await inter.response.send_message(msg, ephemeral=True); return
            names = self.idm.list_here(inter.channel.id, self.mem.get_alias)
            out = " ".join(f"{n}." for n in names) if names else "no one."
            out = self._decorate_unknown_once(inter.user.id, out)
            await inter.response.send_message(out, ephemeral=False)

        @self.tree.command(name="setalias", description="set my alias (public)")
        @app_commands.describe(name="short name (1-2 words)")
        async def setalias(inter: discord.Interaction, name:str):
            if not _allowed(inter):
                msg = self._decorate_unknown_once(inter.user.id, "not allowed in this channel")
                await inter.response.send_message(msg, ephemeral=True); return
            clean = " ".join(name.strip().split()[:2])[:32]
            self.mem.set_alias(inter.user.id, clean)
            await inter.response.send_message("noted.", ephemeral=True)

        @self.tree.command(name="ping", description="latency")
        async def ping(inter: discord.Interaction):
            if not _allowed(inter):
                msg = self._decorate_unknown_once(inter.user.id, "not allowed in this channel")
                await inter.response.send_message(msg, ephemeral=True); return
            await inter.response.send_message(self._decorate_unknown_once(inter.user.id, "pong!"), ephemeral=True)

    async def on_ready(self):
        log.info("logged in as %s", self.user)

    # ----- silence utilities -----
    def is_silenced(self, user_id:int) -> bool:
        unmute = self.silenced_users.get(user_id)
        if not unmute: return False
        if time.time() >= unmute:
            del self.silenced_users[user_id]; return False
        return True
    def silence_user(self, user_id:int):
        self.silenced_users[user_id] = time.time() + SILENCE_DURATION
    def unsilence_user(self, user_id:int):
        if user_id in self.silenced_users: del self.silenced_users[user_id]

    # ----- message handler -----
    async def on_message(self, m: discord.Message):
        if m.author.bot: return

        # DMs (always addressed)
        if isinstance(m.channel, discord.DMChannel):
            self.idm.note_message(0, m.author, m.content or "")
            a = self.idm.alias.get(m.author.id)
            if a and (self.mem.get_alias(m.author.id) or "") != a:
                self.mem.set_alias(m.author.id, a)
            await self._handle_dm(m); return

        # allowed channels only
        if ALLOWED_CHANNELS and str(m.channel.id) not in ALLOWED_CHANNELS: return
        if self.is_silenced(m.author.id): return

        # presence + alias capture (auto-persist)
        self.idm.note_message(m.channel.id, m.author, m.content or "")
        new_alias = self.idm.alias.get(m.author.id)
        if new_alias and (self.mem.get_alias(m.author.id) or "") != new_alias:
            self.mem.set_alias(m.author.id, new_alias)

        low = (m.content or "").strip().lower()
        addressed = self._is_addressed(m)

        # if not addressed: silent store (S3 positive facts only) + passive maybe; then exit
        if not addressed:
            await self._extract_and_store_facts_regex(m.author.id, m.content or "", conservative=True)
            await self._maybe_passive_opinion(m)
            return

        # identity / silence controls (addressed only)
        if any(p in low for p in ("shut up","hush","stop talking","do not respond","do not reply","stfu","be quiet","don't respond","don't reply")):
            msg = self._decorate_unknown_once(m.author.id, "understood. i will be quiet.")
            await self.send_lc(m.channel, msg); self.silence_user(m.author.id); return
        if low.startswith(("resume","speak","start","reply")):
            self.unsilence_user(m.author.id)
            msg = self._decorate_unknown_once(m.author.id, "resumed.")
            await self.send_lc(m.channel, msg); return

        if low in {"who are you","who are you?","what are you","what are you?"}:
            await self.send_lc(m.channel, "ninja"); return
        if low in {"who am i","who am i?"}:
            out = self._answer_memory_direct(m.author.id)
            out = self._decorate_unknown_once(m.author.id, out)
            await self.send_lc(m.channel, out); return
        if any(k in low for k in MEM_Q_PATTERNS["what_like"]["any"]):
            out = self._answer_memory_direct(m.author.id)
            out = self._decorate_unknown_once(m.author.id, out)
            await self.send_lc(m.channel, out); return
        if "who is here" in low:
            names = self.idm.list_here(m.channel.id, self.mem.get_alias)
            out = " ".join(f"{n}." for n in names) if names else "no one."
            out = self._decorate_unknown_once(m.author.id, out)
            await self.send_lc(m.channel, out); return
        if (low.startswith("is ") and low.endswith(" here")) or low.endswith(" here?"):
            name = low[3:].replace(" here","").replace(" here?","").strip(" ?.!")
            yes = self.idm.is_here(m.channel.id, name, self.mem.get_alias)
            msg = self._decorate_unknown_once(m.author.id, "yes." if yes else "no.")
            await self.send_lc(m.channel, msg); return
        if "how many people" in low:
            cnt = len(self.idm.list_here(m.channel.id, self.mem.get_alias))
            msg = self._decorate_unknown_once(m.author.id, f"{cnt}." if cnt>0 else "0.")
            await self.send_lc(m.channel, msg); return
        if low in {"who are you talking to","who are you talking to?"}:
            msg = self._decorate_unknown_once(m.author.id, "i am addressing the channel.")
            await self.send_lc(m.channel, msg); return

        # rate limit
        if not self.rate.allow(m.author.id, m.channel.id): return

        # recall per-user (for general questions)
        recall_snips: List[str] = []
        try:
            qvec = await self.embed(low or "conversation")
            recall_snips = self.mem.recall(m.author.id, qvec, topk=RECALL_TOPK)
        except Exception as e:
            log.warning("recall failed: %s", e)

        # generate (addressed)
        self.mem.add_short(m.channel.id, m.author.id, "user", (m.content or "").strip())
        out = await self.gen_public(m.author.id, (m.content or "").strip(), self.mem.get_short(m.channel.id, m.author.id), recall_snips)
        out = self._decorate_unknown_once(m.author.id, out)
        await self.send_lc(m.channel, out)
        self.mem.add_short(m.channel.id, m.author.id, "assistant", out)

        # store memory sparsely (notes)
        if len(low) >= STORE_MIN_LEN and self.mem.can_store(m.author.id):
            try:
                vec = await self.embed(low)
                self.mem.add_memory(m.author.id, "note", _clamp_text((m.content or "").strip(), EMBED_TRUNCATE), vec)
                self.mem.note_stored(m.author.id)
            except Exception as e:
                log.warning("store memory failed: %s", e)

        # aggressive fact extraction (regex + LLM since addressed)
        await self._extract_and_store_facts_regex(m.author.id, m.content or "", conservative=False)
        await self._extract_and_store_facts_ai(m.author.id, m.content or "")

    # ----- dm mission handler -----
    def _is_refusal(self, txt:str) -> bool:
        t = txt.strip().lower()
        return t in {"no","stop","leave me alone","go away","not now","nah"} or "do not dm" in t or "don't dm" in t
    def _is_probe(self, txt:str) -> bool:
        t = txt.strip().lower()
        probes = ("who are you","who sent","motive","why me","what is this","what are you")
        return any(p in t for p in probes)
    def _is_confused(self, txt:str) -> bool:
        t = txt.strip().lower()
        return t in {"what","huh","?","why","explain"} or (t.endswith("?") and len(t) <= 40)

    async def _handle_dm(self, m: discord.Message):
        row = self.mem.get_agenda(m.author.id)
        if not row: return
        if time.time() - self.mem.last_dm(m.author.id) < DM_MIN_INTERVAL: return

        agenda = row["agenda"]
        steps: List[str] = json.loads(row["steps"])
        idx = int(row["idx"])
        txt = (m.content or "").strip().lower()
        owner_id = int(row["owner_id"] or 0)
        warned = bool(int(row["warned"] or 0))

        # handshake
        if idx == 0:
            if txt.startswith(("y","ready","ok","yes","sure")):
                self.mem.advance(m.author.id); self.mem.touch(m.author.id)
                msg = steps[1] if len(steps) > 1 else "state your aim in one line."
                msg = self._decorate_unknown_once(m.author.id, msg)
                await self.send_lc(m.channel, msg)
            elif self._is_refusal(txt):
                if not warned:
                    msg = self._decorate_unknown_once(m.author.id, "consider your choice. are you certain?")
                    await self.send_lc(m.channel, msg)
                    self.mem.set_warned(m.author.id); self.mem.touch(m.author.id)
                else:
                    msg = self._decorate_unknown_once(m.author.id, "as you wish. i withdraw.")
                    await self.send_lc(m.channel, msg)
                    self.mem.stop_agenda(m.author.id); self.mem.touch(m.author.id)
                    if owner_id: await self.report_owner(owner_id, m.author, "aborted")
            else:
                msg = self._decorate_unknown_once(m.author.id, "say yes when ready.")
                await self.send_lc(m.channel, msg)
            return

        # probes: brief reveal then redirect
        if self._is_probe(txt):
            msg = self._decorate_unknown_once(m.author.id, f"{self.reveal()}")
            await self.send_lc(m.channel, msg); self.mem.touch(m.author.id)
            q = steps[idx] if idx < len(steps) else "return to the task."
            q = self._decorate_unknown_once(m.author.id, q)
            await self.send_lc(m.channel, q); return

        # refusal after start
        if self._is_refusal(txt):
            if not warned:
                msg = self._decorate_unknown_once(m.author.id, "consider your choice. are you certain?")
                await self.send_lc(m.channel, msg); self.mem.set_warned(m.author.id); self.mem.touch(m.author.id); return
            msg = self._decorate_unknown_once(m.author.id, "as you wish. i withdraw.")
            await self.send_lc(m.channel, msg)
            self.mem.stop_agenda(m.author.id); self.mem.touch(m.author.id)
            if owner_id: await self.report_owner(owner_id, m.author, "aborted")
            return

        # confusion → simpler
        if self._is_confused(txt):
            q = steps[idx] if idx < len(steps) else "clarify the last point."
            simpler = await self.rephrase_question(q)
            simpler = self._decorate_unknown_once(m.author.id, simpler)
            await self.send_lc(m.channel, simpler); self.mem.touch(m.author.id); return

        # last planned step?
        last_planned = (idx >= len(steps)-1)
        if last_planned:
            transcript = f"q:{steps[idx] if idx < len(steps) else ''}\na:{txt}"
            result = await self.compress_result(agenda, transcript)
            msg = self._decorate_unknown_once(m.author.id, self.closing())
            await self.send_lc(m.channel, msg)
            self.mem.stop_agenda(m.author.id); self.mem.touch(m.author.id)
            if owner_id: await self.report_owner(owner_id, m.author, result)
            # store mission result
            try:
                vec = await self.embed(f"mission: {agenda} | result: {result}")
                self.mem.add_memory(m.author.id, "event", f"mission result: {result} (agenda: {agenda})", vec)
            except Exception as e:
                log.warning("store result memory failed: %s", e)
            # DM also feeds facts (H2)
            await self._extract_and_store_facts_ai(m.author.id, m.content or "")
            return

        # weak reply → stay, re-ask simpler
        if txt in {"idk","not sure","ok","k","fine","maybe","dunno"} or len(txt) < 2:
            q = await self.rephrase_question(steps[idx])
            q = self._decorate_unknown_once(m.author.id, q)
            await self.send_lc(m.channel, q); self.mem.touch(m.author.id); return

        # normal advance
        self.mem.advance(m.author.id); self.mem.touch(m.author.id)
        nxt = steps[idx+1] if idx+1 < len(steps) else await self.followup_question(agenda, steps[idx], txt)
        nxt = self._decorate_unknown_once(m.author.id, nxt)
        await self.send_lc(m.channel, nxt)
        # DM facts (regex + LLM)
        await self._extract_and_store_facts_regex(m.author.id, m.content or "", conservative=False)
        await self._extract_and_store_facts_ai(m.author.id, m.content or "")

    # ----- background: notify owner if no response in 24h -----
    async def agenda_watchdog(self):
        while True:
            try:
                now = _now()
                rows = self.mem.active_rows()
                for row in rows:
                    uid = int(row["user_id"]); owner_id = int(row["owner_id"] or 0)
                    created = float(row["created"] or 0); last_dm = float(row["last_dm"] or 0)
                    if owner_id and last_dm == 0 and created and (now - created) >= 24*3600:
                        with contextlib.suppress(Exception):
                            owner = await self.fetch_user(owner_id)
                            await self.send_lc(owner, f"no response — target: {uid}")
                        self.mem.stop_agenda(uid)
                await asyncio.sleep(600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"agenda_watchdog error: {e}")
                await asyncio.sleep(60)

    # ----- background: profile refresher (summarize memories to profile) -----
    async def profile_refresher(self):
        while True:
            try:
                await asyncio.sleep(300)
                uids = {uid for (_, uid) in self.mem.short.keys()}
                for uid in uids:
                    if (_now() - self.mem.last_profile_time(uid)) < MEM_SUMMARY_COOLDOWN_S:
                        continue
                    rows = self.mem.conn.execute(
                        "SELECT text FROM memories WHERE user_id=? ORDER BY ts DESC LIMIT 50", (uid,)
                    ).fetchall()
                    if not rows:
                        self.mem.note_profile_time(uid); continue
                    corpus = "\n".join(r["text"] for r in rows)
                    sys = ("summarize stable facts and preferences about this user in <=60 words, "
                           "lowercase, no names not present, no speculation.")
                    try:
                        r = await self.oa.chat.completions.create(
                            model=MODEL,
                            messages=[{"role":"system","content":sys},{"role":"user","content":corpus}],
                            temperature=0.2, stream=False)
                        profile = (r.choices[0].message.content or "").strip().lower()
                        self.mem.set_profile(uid, profile)
                        self.mem.note_profile_time(uid)
                    except Exception as e:
                        log.warning("profile refresh failed: %s", e)
                        self.mem.note_profile_time(uid)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("profile refresher error: %s", e)
                await asyncio.sleep(60)

    async def close(self):
        await self.session.close()
        await super().close()

# ----- run -----
async def main():
    bot = Ninja()
    try:
        await bot.start(DISCORD_TOKEN)
    finally:
        await bot.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
