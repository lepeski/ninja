import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands

from core.assistant import Notification

log = logging.getLogger(__name__)


def should_bot_reply(message: discord.Message, recent_messages: List[Dict], bot_user: Optional[discord.User]) -> bool:
    """Determine whether the bot should reply to a group message."""

    if message.guild is None:
        return True

    content = message.content or ""
    stripped = content.strip()
    lowered = stripped.lower()
    bot_id = bot_user.id if bot_user else None

    # Signal A: direct summons or command prefixes.
    if lowered.startswith("ninja"):
        return True
    if stripped.startswith("!"):
        return True
    if bot_id is not None:
        if any(user.id == bot_id for user in message.mentions):
            return True
        raw_id = str(bot_id)
        if f"<@{raw_id}>" in content or f"<@!{raw_id}>" in content:
            return True

    # Signal B heuristics.
    lowered_no_punct = lowered
    second_person_triggers = [
        "can you",
        "could you",
        "would you",
        "will you",
        "should i",
        "what should i",
        "how do i",
        "do you know",
        "you think",
        "your take",
        "are you",
        "did you",
    ]

    mentions_other = any(
        mention.id != bot_id for mention in message.mentions
    ) if message.mentions else False

    question_without_target = "?" in content and not mentions_other
    if question_without_target:
        return True

    if any(trigger in lowered_no_punct for trigger in second_person_triggers):
        return True

    if recent_messages:
        last_bot_message = next(
            (entry for entry in reversed(recent_messages) if entry.get("from_bot")),
            None,
        )
        if last_bot_message:
            if "you" in lowered_no_punct or "your" in lowered_no_punct:
                return True
            if any(word in lowered_no_punct for word in ["that", "this", "those", "it"]):
                if any(starter in lowered_no_punct for starter in ["what", "why", "how", "when", "where"]):
                    return True

    return False


class DiscordTransport(commands.Bot):
    def __init__(self, assistant, *, guild_id: Optional[int] = None):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        self._primary_prefix = "!"
        super().__init__(command_prefix=self._primary_prefix, intents=intents)
        self.assistant = assistant
        self.guild_id = guild_id
        self._recent_messages: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

    async def setup_hook(self) -> None:
        self.tree.add_command(self._assign_agenda())
        self.tree.add_command(self._stop_agenda())
        if self.guild_id:
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    def _assign_agenda(self) -> app_commands.Command:
        @app_commands.command(name="assignagenda", description="Assign a mission to a user")
        @app_commands.describe(
            user="Target user",
            goal="Mission goal",
            timeout_hours="Hours before the mission expires (optional)",
        )
        async def assignagenda(
            interaction: discord.Interaction,
            user: discord.User,
            goal: str,
            timeout_hours: Optional[float] = None,
        ):
            await interaction.response.defer(ephemeral=True)
            try:
                mission, ack, dm_message = await self.assistant.start_mission(
                    creator_id=f"discord:{interaction.user.id}",
                    target_id=f"discord:{user.id}",
                    objective=goal,
                    timeout_hours=timeout_hours,
                    creator_name=interaction.user.display_name,
                    target_name=user.display_name,
                )
            except PermissionError as exc:
                await interaction.followup.send(str(exc), ephemeral=True)
                return
            dm_failed = False
            try:
                await user.send(dm_message)
            except Exception as exc:
                dm_failed = True
                log.warning("Failed to DM mission to %s: %s", user.id, exc)
            note = " DM delivered." if not dm_failed else " Unable to DM the user."
            await interaction.followup.send(f"{ack}{note}", ephemeral=True)
            await self._deliver_notifications()

        return assignagenda

    def _stop_agenda(self) -> app_commands.Command:
        @app_commands.command(name="stopagenda", description="Cancel a mission by ID")
        @app_commands.describe(mission_id="Mission identifier to cancel")
        async def stopagenda(interaction: discord.Interaction, mission_id: str):
            await interaction.response.defer(ephemeral=True)
            result = self.assistant.cancel_mission(
                creator_id=f"discord:{interaction.user.id}", mission_id=mission_id
            )
            await interaction.followup.send(result, ephemeral=True)
            await self._deliver_notifications()

        return stopagenda

    async def on_ready(self):
        log.info("Discord bot ready as %s", self.user)

    async def on_message(self, message: discord.Message):
        if not message or not message.content:
            return
        if self.user and message.author.id == self.user.id:
            return
        channel_id = str(message.channel.id)
        is_dm = message.guild is None
        content = message.content
        if not is_dm:
            recent = list(self._recent_messages[channel_id])
            if not should_bot_reply(message, recent, self.user):
                self._remember_channel_message(channel_id, str(message.author.id), content, False)
                await self.assistant.observe_message(
                    platform="discord",
                    user_id=str(message.author.id),
                    message=content,
                    username=message.author.display_name,
                    channel_id=channel_id,
                )
                return
            self._remember_channel_message(channel_id, str(message.author.id), content, False)
            content = self._strip_direct_invocation(content, message)
            if not content.strip():
                content = "ninja"
        try:
            result = await self.assistant.handle_message(
                platform="discord",
                user_id=str(message.author.id),
                message=content,
                username=message.author.display_name,
                channel_id=channel_id,
                is_dm=is_dm,
            )
        except Exception as exc:
            log.exception("Assistant error: %s", exc)
            result = "I'm not available right now."
        if result:
            await message.channel.send(result, reference=message if not is_dm else None)
            if not is_dm:
                self._remember_channel_message(
                    channel_id,
                    str(self.user.id) if self.user else "bot",
                    result,
                    True,
                )
        await self._deliver_notifications()

    async def _deliver_notifications(self) -> None:
        notifications = self.assistant.drain_notifications()
        for note in notifications:
            if not isinstance(note, Notification):
                continue
            if note.platform != "discord":
                continue
            try:
                user_obj = await self.fetch_user(int(note.user_id))
                await user_obj.send(note.message)
            except Exception as exc:
                log.warning("Failed to deliver notification to %s: %s", note.user_id, exc)

    def _remember_channel_message(
        self, channel_id: str, author_id: str, content: str, from_bot: bool
    ) -> None:
        if not channel_id:
            return
        history = self._recent_messages[channel_id]
        history.append(
            {
                "author_id": author_id,
                "content": content,
                "from_bot": from_bot,
            }
        )

    def _strip_direct_invocation(self, content: str, message: discord.Message) -> str:
        stripped = content.strip()
        lowered = stripped.lower()
        if lowered.startswith("ninja"):
            trimmed = stripped[len("ninja") :].lstrip(" ,:;-\t")
            return trimmed or "ninja"
        if stripped.startswith(self._primary_prefix):
            trimmed = stripped[len(self._primary_prefix) :].lstrip()
            return trimmed or stripped
        if self.user and any(user.id == self.user.id for user in message.mentions):
            cleaned = content
            for variant in (
                self.user.mention,
                f"<@{self.user.id}>",
                f"<@!{self.user.id}>",
            ):
                if variant in cleaned:
                    cleaned = cleaned.replace(variant, "", 1).strip()
            return cleaned or ""
        return content


async def run_discord_bot(assistant, token: str, guild_id: Optional[int] = None):
    bot = DiscordTransport(assistant, guild_id=guild_id)
    try:
        await bot.start(token)
    finally:
        await bot.close()
