# core/assistant.py
from __future__ import annotations
import asyncio, contextlib, json, logging, math, os, re, random, sqlite3, time, pathlib
from collections import defaultdict, deque
from typing import Dict, Deque, List, Optional, Tuple

import aiohttp
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---- shared config ----
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("assistant")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL           = os.getenv("MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# pacing / limits
USER_RATE_SECONDS = 1
CHANNEL_RATE_SECONDS = 5
CHANNEL_RATE_LIMIT = 10
MAX_HISTORY = 8

# memory knobs
STORE_MIN_LEN = 6
STORE_COOLDOWN_S = 60
RECALL_TOPK = 5
EMBED_TRUNCATE = 800
MEM_SUMMARY_COOLDOWN_S = 6*3600

# passive opinions
PASSIVE_CHANCE = float(os.getenv("PASSIVE_TRIGGER_CHANCE", "0.05"))

UNKNOWN_SUFFIX = "nigga"

MEM_DIR = pathlib.Path(os.getenv("MEM_DIR", "mem"))
MEM_DIR.mkdir(parents=True, exist_ok=True)

MEM_FACT_PATTERNS_POS = [
    r"\bi like ([^.,;]+)", r"\bi love ([^.,;]+)", r"\bi enjoy ([^.,;]+)",
    r"\bmy favorite (?:game|food|thing|song|movie|band|color|sport) is ([^.,;]+)"
]
MEM_FACT_PATTERNS_NEG = [r"\bi (?:hate|dislike) ([^.,;]+)"]

MEM_Q_PATTERNS = {
    "whoami": {"any": ["who am i"]},
    "what_like": {"any": ["what do i like","what do you know about me","what do you remember about me","tell me about myself"]},
}

def _now() -> float: return time.time()
def _clamp_text(s: str, n: int) -> str: return s if len(s) <= n else s[:n]
def _norm(s:str) -> str: return re.sub(r"\s+", " ", (s or "").strip().lower())

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b): return 0.0
    sa = math.sqrt(sum(x*x for x in a)); sb = math.sqrt(sum(x*x for x in b))
    if sa == 0 or sb == 0: return 0.0
    return sum(x*y for x,y in zip(a,b)) / (sa*sb)

# ---------- Memory ----------
class Memory:
    def __init__(self, path="memory.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.short: Dict[Tuple[str,int], Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
        self._init_db()
        self._last_store: Dict[int, float] = {}

    def _init_db(self):
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              ts REAL NOT NULL,
              kind TEXT NOT NULL,
              text TEXT NOT NULL,
              embedding TEXT NOT NULL
            )""")

    def add_short(self, chan_key:str, user_id:int, role:str, content:str):
        self.short[(chan_key,user_id)].append((role, content))

    def get_short(self, chan_key:str, user_id:int):
        return list(self.short[(chan_key,user_id)])

    def add_memory(self, uid:int, kind:str, text:str, embedding:List[float]):
        with self.conn:
            self.conn.execute(
                "INSERT INTO memories(user_id, ts, kind, text, embedding) VALUES(?,?,?,?,?)",
                (uid, _now(), kind, text, json.dumps(embedding))
            )

    def can_store(self, uid:int) -> bool:
        return (_now() - self._last_store.get(uid, 0)) >= STORE_COOLDOWN_S

    def note_stored(self, uid:int):
        self._last_store[uid] = _now()

    def recall(self, uid:int, query_vec:List[float], topk:int=RECALL_TOPK) -> List[str]:
        rows = self.conn.execute(
            "SELECT text, embedding FROM memories WHERE user_id=? ORDER BY ts DESC LIMIT 400", (uid,)
        ).fetchall()
        scored = []
        for r in rows:
            try:
                vec = json.loads(r["embedding"])
                scored.append((cosine(query_vec, vec), r["text"]))
            except:
                pass
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for s,t in scored[:topk] if s > 0.2]

# ---------- Identity ----------
class Identity:
    def __init__(self):
        self.alias = {}
    def note_message(self, chan_key:str, user_id:int, text:str):
        pass

# ---------- Rate limiter ----------
class RateLimiter:
    def __init__(self):
        self.user_ts = {}
        self.chan_ts = defaultdict(deque)

    def allow(self, user_id:int, chan_key:str) -> bool:
        now = time.monotonic()
        t = self.user_ts.get(user_id)
        if t and now - t < USER_RATE_SECONDS:
            return False
        q = self.chan_ts[chan_key]
        while q and now - q[0] > CHANNEL_RATE_SECONDS:
            q.popleft()
        if len(q) >= CHANNEL_RATE_LIMIT:
            return False
        self.user_ts[user_id] = now
        q.append(now)
        return True

# ---------- Assistant (brain) ----------
class Assistant:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise SystemExit("missing OPENAI_API_KEY")
        self.mem = Memory()
        self.rate = RateLimiter()
        self.oa = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.session = aiohttp.ClientSession()

    async def embed(self, text:str) -> List[float]:
        text = _clamp_text(text, EMBED_TRUNCATE)
        r = await self.oa.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return r.data[0].embedding

    async def gen_public(self, text:str, hist: List[Tuple[str,str]]) -> str:
        sys = "calm samurai. respond in <=2 short sentences. lowercase."
        msgs = [{"role":"system","content":sys}]
        for role,cont in hist[-6:]:
            msgs.append({"role":role,"content":cont})
        msgs.append({"role":"user","content":text})
        r = await self.oa.chat.completions.create(model=MODEL, messages=msgs, temperature=0.5)
        return (r.choices[0].message.content or "").strip().lower()

    async def handle_message(self, *, platform:str, channel_id:str, user_id:int, text:str, addressed:bool, is_dm:bool):
        txt = (text or "").strip()
        if not txt:
            return None

        if not (is_dm or addressed):
            return None  # passive ignore for mode A

        if not self.rate.allow(user_id, channel_id):
            return None

        # recall
        recall_snips = []
        try:
            qvec = await self.embed(txt.lower())
            recall_snips = self.mem.recall(user_id, qvec, topk=RECALL_TOPK)
        except:
            pass

        self.mem.add_short(channel_id, user_id, "user", txt)
        out = await self.gen_public(txt, self.mem.get_short(channel_id, user_id))
        self.mem.add_short(channel_id, user_id, "assistant", out)

        if len(txt) >= STORE_MIN_LEN and self.mem.can_store(user_id):
            try:
                vec = await self.embed(txt.lower())
                self.mem.add_memory(user_id, "note", _clamp_text(txt, EMBED_TRUNCATE), vec)
                self.mem.note_stored(user_id)
            except:
                pass

        return out

    async def close(self):
        await self.session.close()
