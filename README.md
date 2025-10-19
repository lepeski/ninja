# Ninja Discord Bot (Python)

A single-file Discord bot that streams GPT responses, keeps short/long-term memory, polls X trending topics, and responds to mentions or slash commands.

## Prerequisites

- Python 3.10 or newer
- Discord bot token and OpenAI API key
- (Optional) X/Twitter developer bearer token for trend triggers
- (Optional) Running [ChromaDB](https://www.trychroma.com/) server for vector storage

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the sample environment file and fill in your secrets:
   ```bash
   cp .env.example .env
   # edit .env with your values
   ```
4. Run the bot:
   ```bash
   python bot.py
   ```

On first run the bot will register slash commands automatically. Watch the console output for any authentication or network errors.

## Troubleshooting

- If the bot prints an error about missing modules, rerun the `pip install` command.
- Without `X_BEARER_TOKEN` the bot simply skips trend polling.
- If ChromaDB is offline the bot falls back to SQLite only and logs a warning.
- Use `/reset` (owner only) to wipe memory if conversations get stuck.

