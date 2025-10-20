import logging
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

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
        @app_commands.describe(user="Target user", goal="Mission goal")
        async def assignagenda(interaction: discord.Interaction, user: discord.User, goal: str):
            await interaction.response.defer(ephemeral=True)
            ack, dm_message = await self.assistant.assign_agenda(
                platform="discord",
                target_user_id=str(user.id),
                target_username=user.display_name,
                goal=goal,
                owner_id=str(interaction.user.id),
            )
            dm_failed = False
            try:
                await user.send(dm_message)
            except Exception as exc:
                dm_failed = True
                log.warning("Failed to DM mission to %s: %s", user.id, exc)
            note = " DM delivered." if not dm_failed else " Unable to DM the user."
            await interaction.followup.send(ack + note, ephemeral=True)

        return assignagenda

    def _stop_agenda(self) -> app_commands.Command:
        @app_commands.command(name="stopagenda", description="Stop a user's mission")
        @app_commands.describe(user="User whose mission should stop")
        async def stopagenda(
            interaction: discord.Interaction, user: Optional[discord.User] = None
        ):
            await interaction.response.defer(ephemeral=True)
            target = user or interaction.user
            result = await self.assistant.stop_agenda(
                platform="discord",
                target_user_id=str(target.id),
            )
            await interaction.followup.send(result, ephemeral=True)

        return stopagenda

    async def on_ready(self):
        log.info("Discord bot ready as %s", self.user)

    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.content:
            return
        channel_id = message.channel.id
        is_dm = message.guild is None
        try:
            result = await self.assistant.handle_message(
                platform="discord",
                user_id=str(message.author.id),
                username=message.author.display_name,
                channel_id=str(channel_id),
                message=message.content,
                is_dm=is_dm,
            )
        except Exception as exc:
            log.exception("Assistant error: %s", exc)
            result = "I can't respond right now."
        if not result:
            return
        reply_text = str(result)
        if reply_text:
            await message.channel.send(reply_text, reference=message if not is_dm else None)


async def run_discord_bot(assistant, token: str, guild_id: Optional[int] = None):
    bot = DiscordTransport(assistant, guild_id=guild_id)
    try:
        await bot.start(token)
    finally:
        await bot.close()
