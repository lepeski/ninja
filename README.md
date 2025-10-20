# Ninja Multi-Platform Assistant

A shared AI assistant that powers both Discord and Telegram transports. The core assistant handles long-term memory, concise
responses, fact extraction, and DM-only missions while each transport deals only with platform plumbing.

## Features

- Shared assistant logic for Discord and Telegram.
- Long-term SQLite + embedding memory scoped by platform/user.
- Fact extraction for likes/dislikes with "who am I" recall answers.
- Suffix for unknown users to confirm identity.
- Agenda (mission) system driven by slash/command handlers, executed only in DMs.
- Concise, direct voice on every reply.
- Archives unsolicited DMs before responding.

## Requirements

- Python 3.10+
- Discord bot token
- Telegram bot token
- OpenAI API key with Chat Completions + Embeddings access

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the environment template and fill in your secrets:
   ```bash
   cp .env.example .env
   ```
2. Update `.env` with your API keys, optional guild ID, and any model overrides.
3. Ensure the `mem/` directory is writable (created automatically on first run).

Environment variables:

- `OPENAI_API_KEY` – OpenAI API key.
- `DISCORD_TOKEN` – Discord bot token.
- `TELEGRAM_TOKEN` – Telegram bot token.
- `MODEL` – Chat model name (default `gpt-4.1-mini`).
- `EMBEDDING_MODEL` – Embedding model name (default `text-embedding-3-small`).
- `MEM_DIR` – Directory for JSON mirrors of user memories.
- `DISCORD_GUILD_ID` – Optional guild ID to scope slash commands.

## Running

Start both transports with a single command:

```bash
python main.py
```

The process blocks until interrupted (Ctrl+C). Slash commands `/assignagenda` and `/stopagenda` are available on Discord. Telegram
users invoke `/assignagenda <goal>` and `/stopagenda` via direct messages only.

## Data Storage

- Long-term memories live in `memory.db` (SQLite) with embeddings per platform/user.
- Human-readable mirrors and facts are stored under `mem/` as JSON files named `{platform}_{user}.json`.
- First-contact private messages are archived under `mem/inbox/` before the assistant replies.

Back up or clear these files to reset the assistant's recollections.
