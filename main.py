import asyncio
import logging
import os
import signal

from dotenv import load_dotenv

from core.assistant import Assistant
from transports.discord_bot import run_discord_bot
from transports.telegram_bot import TelegramTransport


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")

    openai_key = os.getenv("OPENAI_API_KEY")
    discord_token = os.getenv("DISCORD_TOKEN")
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    model = os.getenv("MODEL", "gpt-4.1")
    mem_dir = os.getenv("MEM_DIR", "mem")
    guild_id_raw = os.getenv("DISCORD_GUILD_ID")
    guild_id = int(guild_id_raw) if guild_id_raw and guild_id_raw.isdigit() else None

    if not openai_key or not discord_token or not telegram_token:
        raise SystemExit("Missing required environment variables.")

    assistant = Assistant(
        openai_api_key=openai_key,
        model=model,
        mem_dir=mem_dir,
    )

    telegram_transport = TelegramTransport(assistant, telegram_token)

    stop_event = asyncio.Event()

    def _signal_handler(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop_event.set())

    discord_task = asyncio.create_task(run_discord_bot(assistant, discord_token, guild_id))
    telegram_task = asyncio.create_task(telegram_transport.start())

    await stop_event.wait()

    await telegram_transport.stop()

    discord_task.cancel()
    try:
        await discord_task
    except asyncio.CancelledError:
        pass

    await telegram_task
    await assistant.close()


if __name__ == "__main__":
    asyncio.run(main())
