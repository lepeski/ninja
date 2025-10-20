import asyncio
import logging

from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

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
            reply = await self.assistant.handle_message(
                platform="telegram",
                user_id=str(user.id),
                username=user.full_name or user.username or str(user.id),
                channel_id=str(chat.id),
                message=message.text,
                is_dm=is_dm,
            )
        except Exception as exc:
            log.exception("Assistant error: %s", exc)
            reply = "My words are lost on the wind."
        if reply:
            await message.reply_text(reply)

    async def assign_agenda(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        user = update.effective_user
        if not chat or not user:
            return
        if chat.type != ChatType.PRIVATE:
            return
        goal = " ".join(context.args).strip()
        if not goal:
            await update.effective_message.reply_text("Share the mission goal after /assignagenda.")
            return
        result = await self.assistant.assign_agenda(
            platform="telegram",
            target_user_id=str(user.id),
            target_username=user.full_name or user.username or str(user.id),
            goal=goal,
            owner_id=str(user.id),
        )
        await update.effective_message.reply_text(result)

    async def stop_agenda(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        user = update.effective_user
        if not chat or not user:
            return
        if chat.type != ChatType.PRIVATE:
            return
        result = await self.assistant.stop_agenda(
            platform="telegram",
            target_user_id=str(user.id),
        )
        await update.effective_message.reply_text(result)

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
