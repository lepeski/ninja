# Ninja Multi-Platform Assistant

Ninja is a shared conversational brain that powers both Discord and Telegram. A single OpenAI-driven agent holds context, long-term memory, and missions while the transports only forward messages and commands.

## Highlights

- **One persona, many rooms** – Discord and Telegram reuse the same assistant instance and memory.
- **Prompt-driven behavior** – All tone, mission handling, and conversational rules live in the system prompt so replies feel adaptive instead of scripted.
- **Per-user long-term memory** – Preferences, facts, personality notes, and active missions are stored in SQLite with JSON mirrors in `mem/`.
- **Context recall** – Each reply considers recent channel history and relevant memories before calling the model.
- **Missions as goals** – Agenda assignments become conversational objectives that the model pursues naturally rather than step-by-step scripts.
- **Trigger handling** – In group chats the bot only answers when messages start with `ninja`; direct messages always receive a response.
- **Inbox archiving** – First-contact private messages are archived in `mem/inbox/` before the assistant replies.

## Requirements

- Python 3.10+
- Discord bot token
- Telegram bot token
- OpenAI API key with Chat Completions access

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the environment template and fill in your secrets:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and set:
   - `OPENAI_API_KEY`
   - `DISCORD_TOKEN`
   - `TELEGRAM_TOKEN`
   - Optional overrides: `MODEL`, `MEM_DIR`, `DISCORD_GUILD_ID`
3. Ensure the `mem/` directory remains writable (it is created automatically on first run).

## Running

Start both transports with a single command:

```bash
python main.py
```

Discord exposes `/assignagenda <user> <goal>` and `/stopagenda` slash commands (optionally scoped by `DISCORD_GUILD_ID`).
Telegram offers `/assignagenda <goal>` and `/stopagenda` in DMs only. Each command updates the shared assistant memory so the LLM can pursue missions conversationally.

## Data Layout

- `memory.db` – SQLite store for user profiles, preferences, facts, and mission metadata.
- `mem/` – JSON mirrors and inbox archives for easier inspection.
- `mem/inbox/` – Raw text snapshots of first-contact DMs from new users.

Back up or delete these files to reset the assistant's recollections.
