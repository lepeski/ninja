import asyncio
import logging
from typing import Optional

from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core.assistant import Notification

log = logging.getLogger(__name__)


class TelegramTransport:
    def __init__(self, assistant, token: str):
        self.assistant = assistant
        self.application = Application.builder().token(token).build()
        self._register_handlers()
        self._stop_event = asyncio.Event()

    def _register_handlers(self):
        self.application.add_handler(CommandHandler("assignagenda", self.assign_agenda))
        self.application.add_handler(CommandHandler("stopagenda", self.stop_agenda))
        self.application.add_handler(
            MessageHandler(filters.TEXT & (~filters.COMMAND), self.handle_message)
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.effective_message
        user = update.effective_user
        chat = update.effective_chat
        if not message or not message.text or not user or user.is_bot or not chat:
            return
        is_dm = chat.type == ChatType.PRIVATE
        try:
            result = await self.assistant.handle_message(
                platform="telegram",
                user_id=str(user.id),
                message=message.text,
                username=user.full_name or user.username or str(user.id),
                channel_id=str(chat.id),
                is_dm=is_dm,
            )
        except Exception as exc:
            log.exception("Assistant error: %s", exc)
            result = "I'm not available right now."
        if result:
            await message.reply_text(result)
        await self._deliver_notifications(context)

    async def assign_agenda(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        user = update.effective_user
        if not chat or not user:
            return
        if chat.type != ChatType.PRIVATE:
            return
        goal = " ".join(context.args).strip()
        if not goal:
            await update.effective_message.reply_text(
                "Provide a mission goal after /assignagenda."
            )
            return
        timeout_arg: Optional[float] = None
        if context.args and context.args[-1].replace(".", "", 1).isdigit():
            try:
                timeout_arg = float(context.args[-1])
                goal = " ".join(context.args[:-1]).strip()
            except ValueError:
                timeout_arg = None
        mission, ack, dm_message = await self.assistant.start_mission(
            creator_id=f"telegram:{user.id}",
            target_id=f"telegram:{user.id}",
            objective=goal,
            timeout_hours=timeout_arg,
            target_name=user.full_name or user.username or str(user.id),
        )
        await context.bot.send_message(chat_id=user.id, text=dm_message)
        await update.effective_message.reply_text(ack)
        await self._deliver_notifications(context)

    async def stop_agenda(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        user = update.effective_user
        if not chat or not user:
            return
        if chat.type != ChatType.PRIVATE:
            return
        if not context.args:
            await update.effective_message.reply_text("Provide the mission ID to cancel.")
            return
        mission_id = context.args[0].strip()
        result = self.assistant.cancel_mission(
            creator_id=f"telegram:{user.id}", mission_id=mission_id
        )
        await update.effective_message.reply_text(result)
        await self._deliver_notifications(context)

    async def _deliver_notifications(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        notifications = self.assistant.drain_notifications()
        for note in notifications:
            if not isinstance(note, Notification):
                continue
            if note.platform != "telegram":
                continue
            try:
                await context.bot.send_message(chat_id=int(note.user_id), text=note.message)
            except Exception as exc:
                log.warning("Failed to deliver notification to %s: %s", note.user_id, exc)

    async def start(self):
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        try:
            await self._stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()

    async def stop(self):
        self._stop_event.set()
