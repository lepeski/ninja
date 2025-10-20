import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from core.assistant import Notification

log = logging.getLogger(__name__)


class DiscordTransport(commands.Bot):
    def __init__(self, assistant, *, guild_id: Optional[int] = None):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix="!", intents=intents)
        self.assistant = assistant
        self.guild_id = guild_id

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
            mission, ack, dm_message = await self.assistant.start_mission(
                creator_id=f"discord:{interaction.user.id}",
                target_id=f"discord:{user.id}",
                objective=goal,
                timeout_hours=timeout_hours,
                target_name=user.display_name,
            )
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
        if message.author.bot or not message.content:
            return
        channel_id = str(message.channel.id)
        is_dm = message.guild is None
        try:
            result = await self.assistant.handle_message(
                platform="discord",
                user_id=str(message.author.id),
                message=message.content,
                username=message.author.display_name,
                channel_id=channel_id,
                is_dm=is_dm,
            )
        except Exception as exc:
            log.exception("Assistant error: %s", exc)
            result = "I'm not available right now."
        if result:
            await message.channel.send(result, reference=message if not is_dm else None)
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


async def run_discord_bot(assistant, token: str, guild_id: Optional[int] = None):
    bot = DiscordTransport(assistant, guild_id=guild_id)
    try:
        await bot.start(token)
    finally:
        await bot.close()
