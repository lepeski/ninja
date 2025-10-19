#!/usr/bin/env python3
# ninja discord bot — final transparent version (concise public chat + dm agendas)
from __future__ import annotations
import asyncio, contextlib, json, logging, os, sqlite3, time
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

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4.1-mini")

# owner ids (comma separated) and allowed channels optional
OWNER_IDS = {int(x) for x in os.getenv("OWNER_IDS", "").split(",") if x.strip().isdigit()}
ALLOWED_CHANNELS = {x.strip() for x in os.getenv("ALLOW_CHANNELS", "").split(",") if x.strip()}

# guild id for guild-only slash commands (development)
GUILD_ID = 465454837400731648  # provided

if not DISCORD_TOKEN or not OPENAI_API_KEY:
    raise SystemExit("missing DISCORD_TOKEN or OPENAI_API_KEY")

# pacing / limits
USER_RATE_SECONDS = 1
CHANNEL_RATE_SECONDS = 5
CHANNEL_RATE_LIMIT = 10
DM_MIN_INTERVAL = 0.0
MAX_HISTORY = 8
SILENCE_DURATION = 600  # 10 minutes

# ----- memory (sqlite lightweight) -----
class Memory:
    def __init__(self, path="memory.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.short: Dict[int, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
        self._init_db()
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
              created REAL NOT NULL DEFAULT 0
            )""")
    def add_short(self, cid:int, role:str, content:str):
        self.short[cid].append((role, content))
    def get_short(self, cid:int):
        return list(self.short[cid])
    def upsert_agenda(self, target_id:int, owner_id:int, agenda:str, steps:List[str]):
        now = time.time()
        with self.conn:
            self.conn.execute("""
            INSERT INTO agendas(user_id, agenda, steps, idx, active, last_dm, owner_id, created)
            VALUES(?, ?, ?, 0, 1, 0, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              agenda=excluded.agenda, steps=excluded.steps, idx=0, active=1, last_dm=0, owner_id=excluded.owner_id, created=excluded.created
            """, (target_id, agenda, json.dumps(steps), owner_id, now))
    def get_agenda(self, uid:int):
        return self.conn.execute("SELECT * FROM agendas WHERE user_id=? AND active=1", (uid,)).fetchone()
    def stop_agenda(self, uid:int):
        with self.conn:
            self.conn.execute("UPDATE agendas SET active=0 WHERE user_id=?", (uid,))
    def next_step(self, uid:int) -> Optional[str]:
        row = self.get_agenda(uid)
        if not row: return None
        steps = json.loads(row["steps"])
        idx = int(row["idx"])
        return steps[idx] if idx < len(steps) else None
    def advance(self, uid:int):
        with self.conn:
            self.conn.execute("UPDATE agendas SET idx=idx+1 WHERE user_id=?", (uid,))
    def touch(self, uid:int):
        with self.conn:
            self.conn.execute("UPDATE agendas SET last_dm=? WHERE user_id=?", (time.time(), uid))
    def last_dm(self, uid:int) -> float:
        row = self.get_agenda(uid)
        return float(row["last_dm"] or 0) if row else 0.0
    def active_rows(self):
        return self.conn.execute("SELECT * FROM agendas WHERE active=1").fetchall()

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
        self.user_ts[user_id] = now
        q.append(now)
        return True

# ----- bot -----
class Ninja(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="/", intents=intents)
        self.mem = Memory()
        self.rate = RateLimiter()
        self.oa = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.session = aiohttp.ClientSession()
        self.silenced_users: Dict[int, float] = {}  # user_id -> unmute_ts

    # lowercase send helper
    async def send_lc(self, dest: discord.abc.Messageable, text: str):
        txt = (text or "").strip().lower()
        if not txt: return
        with contextlib.suppress(Exception):
            await dest.send(txt)

    # public generation (fast single-shot)
    async def gen_public(self, user: discord.User, text: str, history: List[Tuple[str,str]]) -> str:
        sys = "you are concise and helpful. lowercase only. no filler. answer directly; if unclear ask one brief question."
        msgs = [{"role":"system","content":sys}]
        for role, cont in history[-6:]:
            msgs.append({"role": role, "content": cont})
        msgs.append({"role":"user","content": text})
        try:
            r = await self.oa.chat.completions.create(model=MODEL, messages=msgs, temperature=0.3, stream=False)
            out = (r.choices[0].message.content or "").strip().lower()
            return out or "ok."
        except Exception as e:
            log.warning("public gen failed: %s", e)
            return "sorry."

    # planner for agenda steps (actionable)
    async def plan_steps(self, agenda: str) -> List[str]:
        sys = "produce 2-4 ultra-brief actionable steps, <=8 words each, lowercase, no fluff."
        try:
            r = await self.oa.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":sys},{"role":"user","content":agenda}],
                temperature=0.2, stream=False)
            text = (r.choices[0].message.content or "").strip().lower()
            lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
            steps = [ln for ln in lines if ln][:4]
            return steps if steps else [agenda.strip().lower()]
        except Exception as e:
            log.warning("planner failed: %s", e)
            return [agenda.strip().lower()]

    # dm phrases (transparent)
    def opener(self) -> str:
        return "my brothers have sent me. we relay your answers back to them. are you ready?"
    def reassure(self) -> str:
        return "calm. answer simply."
    def clarify(self) -> str:
        return "one exact answer."
    def closing(self) -> str:
        return "thx. adios ^^/"

    async def report_owner(self, owner_id:int, target:discord.User, result:str):
        try:
            owner = await self.fetch_user(owner_id)
            await self.send_lc(owner, f"mission complete — target: {target.display_name} — result: {result}")
        except Exception as e:
            log.warning(f"report_owner failed: {e}")

    # ----- events -----
    async def setup_hook(self):
        self.setup_cmds()
        guild = discord.Object(id=GUILD_ID)
        await self.tree.sync(guild=guild)
        log.info(f"slash commands synced to guild {GUILD_ID}")
        asyncio.create_task(self.agenda_watchdog())

    def setup_cmds(self):
        def _allowed(inter: discord.Interaction) -> bool:
            if not ALLOWED_CHANNELS: return True
            # allow only in explicitly listed channels (A1 strict)
            ch_id = getattr(inter.channel, "id", None)
            return ch_id is not None and str(ch_id) in ALLOWED_CHANNELS

        @self.tree.command(name="assignagenda", description="start a concise quest (owner only)")
        @app_commands.describe(user="target user", agenda="short goal text")
        async def assignagenda(inter: discord.Interaction, user: discord.User, agenda: str):
            if not _allowed(inter):
                await inter.response.send_message("not allowed in this channel", ephemeral=True); return
            if OWNER_IDS and inter.user.id not in OWNER_IDS:
                await inter.response.send_message("owner only", ephemeral=True); return
            await inter.response.defer(ephemeral=True)
            steps = await self.plan_steps(agenda)
            steps.insert(0, self.opener())
            self.mem.upsert_agenda(user.id, inter.user.id, agenda, steps)
            await self.send_lc(user, steps[0])
            await inter.followup.send(f"quest begun for {user.display_name}", ephemeral=True)

        @self.tree.command(name="stopagenda", description="stop quest (owner only)")
        async def stopagenda(inter: discord.Interaction, user: discord.User):
            if not _allowed(inter):
                await inter.response.send_message("not allowed in this channel", ephemeral=True); return
            if OWNER_IDS and inter.user.id not in OWNER_IDS:
                await inter.response.send_message("owner only", ephemeral=True); return
            self.mem.stop_agenda(user.id)
            await inter.response.send_message("stopped", ephemeral=True)

        @self.tree.command(name="ping", description="latency")
        async def ping(inter: discord.Interaction):
            if not _allowed(inter):
                await inter.response.send_message("not allowed in this channel", ephemeral=True); return
            await inter.response.send_message("pong!", ephemeral=True)

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

    # ----- message handling -----
    async def on_message(self, m: discord.Message):
        if m.author.bot: return

        # DMs -> agenda flow
        if isinstance(m.channel, discord.DMChannel):
            await self._handle_dm(m); return

        # public: obey allowed channels if set
        if ALLOWED_CHANNELS and str(m.channel.id) not in ALLOWED_CHANNELS:
            return

        # check silence per-user
        if self.is_silenced(m.author.id):
            return

        # detect silence phrases
        low = (m.content or "").strip().lower()
        silence_triggers = ("shut up","hush","stop talking","do not respond","do not reply","stfu","be quiet","don't respond","don't reply")
        if any(p in low for p in silence_triggers):
            await self.send_lc(m.channel, "understood. i will be quiet.")
            self.silence_user(m.author.id)
            return

        # resume triggers
        resume_triggers = ("resume","speak","start","reply")
        if any(low.startswith(r) for r in resume_triggers):
            self.unsilence_user(m.author.id)
            await self.send_lc(m.channel, "resumed.")
            return

        # rate limit
        if not self.rate.allow(m.author.id, m.channel.id):
            return

        # add short history and generate
        self.mem.add_short(m.channel.id, "user", m.content.strip())
        out = await self.gen_public(m.author, m.content.strip(), self.mem.get_short(m.channel.id))
        await self.send_lc(m.channel, out)
        self.mem.add_short(m.channel.id, "assistant", out)

    # ----- dm agenda handler -----
    async def _handle_dm(self, m: discord.Message):
        row = self.mem.get_agenda(m.author.id)
        if not row: return
        if time.time() - self.mem.last_dm(m.author.id) < DM_MIN_INTERVAL: return

        txt = (m.content or "").strip().lower()
        if any(p in txt for p in ("stop","leave me alone","go away","do not dm","stop dm")):
            await self.send_lc(m.channel, "as you wish. we withdraw.")
            self.mem.stop_agenda(m.author.id); self.mem.touch(m.author.id); return

        steps = json.loads(row["steps"])
        idx = int(row["idx"])
        owner_id = int(row["owner_id"] or 0)

        # opener (idx 0): wait for yes-ish
        if idx == 0:
            if txt.startswith("y"):
                self.mem.advance(m.author.id); self.mem.touch(m.author.id)
                if len(steps) > 1:
                    await self.send_lc(m.channel, steps[1])
                else:
                    await self.send_lc(m.channel, "proceed.")
            else:
                await self.send_lc(m.channel, "say yes to begin.")
            return

        # confusion -> brief reassure
        if txt in ("what","huh","?","why","explain") or ("?" in txt and len(txt) < 40):
            await self.send_lc(m.channel, self.reassure()); return

        # final step check (single-word result)
        last_step = (idx >= len(steps)-1)
        if last_step:
            parts = txt.split()
            if len(parts) != 1:
                await self.send_lc(m.channel, self.clarify()); return
            result = parts[0]
            await self.send_lc(m.channel, self.closing())
            self.mem.stop_agenda(m.author.id); self.mem.touch(m.author.id)
            if owner_id:
                await self.report_owner(owner_id, m.author, result)
            return

        # otherwise advance and send next
        self.mem.advance(m.author.id); self.mem.touch(m.author.id)
        nxt = steps[idx+1] if idx+1 < len(steps) else steps[idx]
        await self.send_lc(m.channel, nxt)

    # ----- owner notification if no response in 24h (B2) -----
    async def agenda_watchdog(self):
        while True:
            try:
                now = time.time()
                for row in self.mem.active_rows():
                    uid = int(row["user_id"])
                    owner_id = int(row["owner_id"] or 0)
                    created = float(row["created"] or 0)
                    last_dm = float(row["last_dm"] or 0)
                    if owner_id and last_dm == 0 and created and (now - created) >= 24*3600:
                        # notify owner
                        with contextlib.suppress(Exception):
                            owner = await self.fetch_user(owner_id)
                            await self.send_lc(owner, f"no response — target: {uid}")
                        # stop quest to avoid repeated pings
                        self.mem.stop_agenda(uid)
                await asyncio.sleep(600)  # check every 10 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"agenda_watchdog error: {e}")
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
