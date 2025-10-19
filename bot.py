#!/usr/bin/env python3
"""Discord GPT bot implemented as a single Python file."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import signal
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except Exception:  # pragma: no cover - optional dependency at runtime
    chromadb = None  # type: ignore
    Collection = Any  # type: ignore


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("ninja_bot")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful Discord assistant.")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "8"))
MODEL = os.getenv("MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
TREND_WOEID = os.getenv("TREND_WOEID", "1")
TREND_UPDATE_INTERVAL = os.getenv("TREND_UPDATE_INTERVAL", "15m")
TREND_TRIGGER_CHANCE = float(os.getenv("TREND_TRIGGER_CHANCE", "0.05"))
TREND_KEYWORD_CHANCE = float(os.getenv("TREND_KEYWORD_CHANCE", "0.2"))
ALLOW_CHANNELS = {
    channel.strip()
    for channel in os.getenv("ALLOW_CHANNELS", "").split(",")
    if channel.strip()
}
OWNER_IDS = {
    int(owner.strip())
    for owner in os.getenv("OWNER_IDS", "").split(",")
    if owner.strip().isdigit()
}
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
CHROMA_URL = os.getenv("CHROMA_URL")

if not DISCORD_TOKEN or not OPENAI_API_KEY:
    logger.error("Both DISCORD_TOKEN and OPENAI_API_KEY must be configured in the environment.")
    raise SystemExit(1)

BrevityInstructions = (
    "Use the fewest words possible even if harder to understand. "
    "Fragments allowed. Always prioritize brevity. Drop filler, intros, and signoffs. "
    "Extremely brief like mysterious wise ninja."
)

MAX_DISCORD_MESSAGE_CHARS = 1900
SIMILAR_MEMORY_LIMIT = 5
SIMILARITY_THRESHOLD = 0.3
PASSIVE_RESPONSE_CHANCE = float(
    os.getenv("PASSIVE_TRIGGER_CHANCE", str(TREND_TRIGGER_CHANCE or 0.05))
)
USER_RATE_SECONDS = 10
CHANNEL_RATE_SECONDS = 30
CHANNEL_RATE_LIMIT = 5
HISTORY_MAX_AGE = timedelta(days=30)


class RateLimiter:
    def __init__(self) -> None:
        self._user_times: Dict[int, float] = {}
        self._channel_times: Dict[int, Deque[float]] = defaultdict(deque)

    def allow(self, user_id: int, channel_id: int) -> bool:
        now = time.monotonic()
        last_user_time = self._user_times.get(user_id)
        if last_user_time and now - last_user_time < USER_RATE_SECONDS:
            return False
        timestamps = self._channel_times[channel_id]
        while timestamps and now - timestamps[0] > CHANNEL_RATE_SECONDS:
            timestamps.popleft()
        if len(timestamps) >= CHANNEL_RATE_LIMIT:
            return False
        self._user_times[user_id] = now
        timestamps.append(now)
        return True


class MemoryManager:
    def __init__(self, database_path: str = "memory.db") -> None:
        self._conn = sqlite3.connect(database_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()
        self._short_term: Dict[int, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY)
        )
        self._chroma_collection: Optional[Collection] = None
        if CHROMA_URL and chromadb is not None:
            self._configure_chroma(CHROMA_URL)
        else:
            if CHROMA_URL and chromadb is None:
                logger.warning("chromadb package not available; long-term embeddings disabled.")

    def _setup(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER,
                    user_id INTEGER,
                    role TEXT,
                    content TEXT,
                    embedding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id INTEGER PRIMARY KEY,
                    opted_in INTEGER NOT NULL DEFAULT 1
                )
                """
            )

    def _configure_chroma(self, url: str) -> None:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8000
        try:
            client = chromadb.HttpClient(host=host, port=port)  # type: ignore[arg-type]
            self._chroma_collection = client.get_or_create_collection("discord_memory")
            logger.info("Connected to ChromaDB at %s", url)
        except Exception as exc:  # pragma: no cover - runtime connection failure
            logger.warning("Failed to connect to ChromaDB (%s); falling back to SQLite only.", exc)
            self._chroma_collection = None

    def short_term_append(self, channel_id: int, role: str, content: str) -> None:
        self._short_term[channel_id].append((role, content))

    def short_term_get(self, channel_id: int) -> List[Tuple[str, str]]:
        return list(self._short_term[channel_id])

    def short_term_clear(self) -> None:
        self._short_term.clear()

    def set_opt_status(self, user_id: int, opted_in: bool) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO user_preferences(user_id, opted_in) VALUES(?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET opted_in=excluded.opted_in",
                (user_id, 1 if opted_in else 0),
            )

    def is_opted_in(self, user_id: int) -> bool:
        row = self._conn.execute(
            "SELECT opted_in FROM user_preferences WHERE user_id=?", (user_id,)
        ).fetchone()
        if row is None:
            return True
        return bool(row[0])

    def store_message(
        self,
        channel_id: int,
        user_id: int,
        role: str,
        content: str,
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        embedding_json = json.dumps(list(embedding)) if embedding is not None else None
        created_at = datetime.utcnow().isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT INTO messages(channel_id, user_id, role, content, embedding, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (channel_id, user_id, role, content, embedding_json, created_at),
            )
        if self._chroma_collection is not None and embedding is not None:
            try:
                metadata = {
                    "channel_id": str(channel_id),
                    "user_id": str(user_id),
                    "role": role,
                    "created_at": created_at,
                }
                self._chroma_collection.add(
                    embeddings=[list(map(float, embedding))],
                    documents=[content],
                    metadatas=[metadata],
                    ids=[f"{channel_id}:{user_id}:{time.time_ns()}"],
                )
            except Exception as exc:  # pragma: no cover - runtime failure only
                logger.warning("Failed to add embedding to ChromaDB: %s", exc)

    def prune_old_messages(self) -> None:
        cutoff = datetime.utcnow() - HISTORY_MAX_AGE
        with self._conn:
            self._conn.execute("DELETE FROM messages WHERE created_at < ?", (cutoff.isoformat(),))
        if self._chroma_collection is not None:
            try:
                self._chroma_collection.delete(
                    where={"created_at": {"$lt": cutoff.isoformat()}}
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed pruning Chroma history: %s", exc)

    def fetch_similar(
        self,
        channel_id: int,
        embedding: Sequence[float],
        limit: int = SIMILAR_MEMORY_LIMIT,
    ) -> List[Tuple[str, str]]:
        if self._chroma_collection is not None:
            try:
                results = self._chroma_collection.query(
                    query_embeddings=[list(map(float, embedding))],
                    n_results=limit,
                    where={"channel_id": str(channel_id)},
                )
                docs = results.get("documents") or [[]]
                distances = results.get("distances") or [[]]
                combined = []
                for text, distance in zip(docs[0], distances[0]):
                    similarity = 1 - float(distance)
                    if similarity >= SIMILARITY_THRESHOLD:
                        combined.append(("system", f"Relevant memory: {text}"))
                return combined
            except Exception as exc:  # pragma: no cover
                logger.warning("ChromaDB query failed: %s", exc)
        rows = self._conn.execute(
            "SELECT content, embedding FROM messages WHERE channel_id=? ORDER BY created_at DESC LIMIT ?",
            (channel_id, limit * 4),
        ).fetchall()
        combined: List[Tuple[str, str]] = []
        for row in rows:
            stored_embedding = row[1]
            if not stored_embedding:
                continue
            try:
                stored_vec = json.loads(stored_embedding)
            except json.JSONDecodeError:
                continue
            similarity = cosine_similarity(stored_vec, embedding)
            if similarity >= SIMILARITY_THRESHOLD:
                combined.append(("system", f"Relevant memory: {row[0]}"))
            if len(combined) >= limit:
                break
        return combined

    def reset(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM messages")
        self.short_term_clear()
        if self._chroma_collection is not None:
            try:
                self._chroma_collection.delete(where={})
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed clearing Chroma collection: %s", exc)


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TrendManager:
    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session
        self._keywords: List[str] = []
        self._task: Optional[asyncio.Task[None]] = None
        self._interval = parse_interval_seconds(TREND_UPDATE_INTERVAL)

    def keywords(self) -> List[str]:
        return self._keywords

    async def start(self) -> None:
        if not X_BEARER_TOKEN:
            logger.info("X_BEARER_TOKEN not set; trending keywords disabled.")
            return
        if self._task is None:
            self._task = asyncio.create_task(self._runner())

    async def _runner(self) -> None:
        while True:
            try:
                await self._update()
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed updating trends: %s", exc)
            await asyncio.sleep(self._interval)

    async def _update(self) -> None:
        if not X_BEARER_TOKEN:
            return
        url = f"https://api.x.com/1.1/trends/place.json?id={TREND_WOEID}"
        headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
        async with self._session.get(url, headers=headers, timeout=30) as resp:
            if resp.status != 200:
                logger.warning("Trend fetch failed with status %s", resp.status)
                return
            payload = await resp.json()
        if not payload:
            return
        trends = payload[0].get("trends", [])
        top = [trend.get("name", "") for trend in trends[:10] if trend.get("name")]
        self._keywords = top
        logger.info("Updated trend keywords: %s", ", ".join(top))

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


def parse_interval_seconds(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("ms"):
        base = float(value[:-2]) / 1000.0
    elif value.endswith("s"):
        base = float(value[:-1])
    elif value.endswith("m"):
        base = float(value[:-1]) * 60
    elif value.endswith("h"):
        base = float(value[:-1]) * 3600
    else:
        base = float(value)
    return max(5, int(base))


class NinjaBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="/", intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.rate_limiter = RateLimiter()
        self.memory = MemoryManager()
        self.openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.session = aiohttp.ClientSession()
        self.trends = TrendManager(self.session)
        self.add_app_commands()

    def add_app_commands(self) -> None:
        @self.tree.command(name="ping", description="Check the bot latency")
        async def ping(interaction: discord.Interaction) -> None:
            await interaction.response.send_message("Pong!", ephemeral=True)

        @self.tree.command(name="reset", description="Clear conversation memory")
        async def reset(interaction: discord.Interaction) -> None:
            if not self._is_owner(interaction.user.id):
                await interaction.response.send_message(
                    "Only owners may reset the memory.", ephemeral=True
                )
                return
            self.memory.reset()
            await interaction.response.send_message("Memory cleared.", ephemeral=True)

        @self.tree.command(name="ask", description="Ask the assistant a question")
        @app_commands.describe(prompt="The prompt to send to the assistant")
        async def ask(interaction: discord.Interaction, prompt: str) -> None:
            await interaction.response.defer()
            response = await self.generate_reply(
                channel=interaction.channel,  # type: ignore[arg-type]
                author=interaction.user,
                prompt=prompt,
                force=True,
            )
            if response is None:
                await interaction.followup.send("Unable to respond right now.", ephemeral=True)

        @self.tree.command(name="optin", description="Enable passive replies for a user")
        @app_commands.describe(user="User to opt in; defaults to yourself")
        async def optin(
            interaction: discord.Interaction, user: Optional[discord.User] = None
        ) -> None:
            if not self._is_owner(interaction.user.id):
                await interaction.response.send_message(
                    "Only owners may change opt-in status.", ephemeral=True
                )
                return
            target = user or interaction.user
            self.memory.set_opt_status(target.id, True)
            await interaction.response.send_message(
                f"{target.display_name} opted in to passive replies.", ephemeral=True
            )

        @self.tree.command(name="optout", description="Disable passive replies for a user")
        @app_commands.describe(user="User to opt out; defaults to yourself")
        async def optout(
            interaction: discord.Interaction, user: Optional[discord.User] = None
        ) -> None:
            if not self._is_owner(interaction.user.id):
                await interaction.response.send_message(
                    "Only owners may change opt-in status.", ephemeral=True
                )
                return
            target = user or interaction.user
            self.memory.set_opt_status(target.id, False)
            await interaction.response.send_message(
                f"{target.display_name} opted out of passive replies.", ephemeral=True
            )

    async def setup_hook(self) -> None:
        await self.tree.sync()
        await self.trends.start()
        logger.info("Slash commands registered.")

    def _is_owner(self, user_id: int) -> bool:
        if not OWNER_IDS:
            return True
        return user_id in OWNER_IDS

    async def on_ready(self) -> None:
        logger.info("Logged in as %s", self.user)
        self.memory.prune_old_messages()

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if message.guild is None:
            return
        if ALLOW_CHANNELS and str(message.channel.id) not in ALLOW_CHANNELS:
            return
        if not message.content or len(message.content) > 4000:
            return
        if not self.memory.is_opted_in(message.author.id):
            return

        content = message.content.strip()
        mention = self.user in message.mentions if self.user else False
        if mention:
            if not self.rate_limiter.allow(message.author.id, message.channel.id):
                await self.remember_only(message, content)
                return
            content = content.replace(self.user.mention, "").strip()
            if not content:
                content = message.content.strip()
        else:
            if not self.rate_limiter.allow(message.author.id, message.channel.id):
                return
            triggered = False
            if random.random() <= PASSIVE_RESPONSE_CHANCE:
                triggered = True
            else:
                lowered = content.lower()
                for keyword in self.trends.keywords():
                    if keyword.lower() in lowered and random.random() <= TREND_KEYWORD_CHANCE:
                        triggered = True
                        break
            if not triggered:
                await self.remember_only(message, content)
                return

        await self.generate_reply(
            channel=message.channel,
            author=message.author,
            prompt=content,
            force=True,
        )

    async def remember_only(self, message: discord.Message, content: str) -> None:
        embedding = await self.embed_text(content)
        self.memory.store_message(
            channel_id=message.channel.id,
            user_id=message.author.id,
            role="user",
            content=content,
            embedding=embedding,
        )
        self.memory.short_term_append(message.channel.id, "user", content)

    async def embed_text(self, text: str) -> Optional[List[float]]:
        try:
            response = await self.openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            return list(response.data[0].embedding)
        except Exception as exc:  # pragma: no cover
            logger.warning("Embedding request failed: %s", exc)
            return None

    async def build_prompt_messages(
        self,
        channel: discord.abc.MessageableChannel,
        author: discord.abc.User,
        prompt: str,
        embedding: Optional[List[float]],
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        channel_prompt = os.getenv(f"SYSTEM_PROMPT_{channel.id}")
        system_text = SYSTEM_PROMPT
        if channel_prompt:
            system_text = channel_prompt
        trend_text = ", ".join(self.trends.keywords())
        if trend_text:
            system_text += f"\nTrending keywords: {trend_text}"
        system_text += "\n" + BrevityInstructions
        messages.append({"role": "system", "content": system_text})
        for role, content in self.memory.short_term_get(channel.id):
            messages.append({"role": role, "content": content})
        if embedding is not None:
            for role, content in self.memory.fetch_similar(channel.id, embedding):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def stream_completion(self, messages: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        try:
            stream = await self.openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                stream=True,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Chat completion request failed: %s", exc)
            return
        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content_piece = None
            if isinstance(delta, dict):
                content_piece = delta.get("content")
            else:
                content_piece = getattr(delta, "content", None)
            if content_piece:
                if isinstance(content_piece, list):
                    content_piece = "".join(str(part) for part in content_piece)
                yield content_piece

    async def generate_reply(
        self,
        channel: Optional[discord.abc.MessageableChannel],
        author: discord.abc.User,
        prompt: str,
        force: bool = False,
    ) -> Optional[str]:
        if channel is None:
            return None
        if not isinstance(channel, discord.TextChannel):
            return None
        if not force and not self.rate_limiter.allow(author.id, channel.id):
            return None
        embedding = await self.embed_text(prompt)
        messages = await self.build_prompt_messages(channel, author, prompt, embedding)
        buffer = ""
        last_edited_len = 0
        async with channel.typing():
            sent = await channel.send("...")
            async for piece in self.stream_completion(messages):
                buffer += piece
                if len(buffer) - last_edited_len >= 20:
                    await sent.edit(content=buffer[-MAX_DISCORD_MESSAGE_CHARS:])
                    last_edited_len = len(buffer)
            if buffer:
                await sent.edit(content=buffer[:MAX_DISCORD_MESSAGE_CHARS])
            else:
                await sent.edit(content="(no response)")
        if buffer:
            chunks = split_chunks(buffer, MAX_DISCORD_MESSAGE_CHARS)
            await sent.edit(content=chunks[0])
            for chunk in chunks[1:]:
                await channel.send(chunk)
        self.memory.store_message(
            channel_id=channel.id,
            user_id=author.id,
            role="user",
            content=prompt,
            embedding=embedding,
        )
        self.memory.short_term_append(channel.id, "user", prompt)
        if buffer:
            bot_embedding = await self.embed_text(buffer)
            self.memory.store_message(
                channel_id=channel.id,
                user_id=self.user.id if self.user else 0,
                role="assistant",
                content=buffer,
                embedding=bot_embedding,
            )
            self.memory.short_term_append(channel.id, "assistant", buffer)
        return buffer

    async def close(self) -> None:
        await self.trends.stop()
        await self.session.close()
        await super().close()


def split_chunks(text: str, max_len: int) -> List[str]:
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]


async def main() -> None:
    bot = NinjaBot()
    loop = asyncio.get_running_loop()

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        if not stop_event.is_set():
            stop_event.set()
            loop.call_soon_threadsafe(lambda: asyncio.create_task(bot.close()))

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:  # pragma: no cover - Windows
            signal.signal(
                sig,
                lambda *_: loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(bot.close())
                ),
            )

    async def runner() -> None:
        try:
            await bot.start(DISCORD_TOKEN)
        finally:
            stop_event.set()

    await asyncio.gather(runner(), stop_event.wait())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:  # pragma: no cover
        pass
