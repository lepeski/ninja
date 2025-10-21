# Ninja Multi-Platform Assistant

Ninja is a shared conversational brain that powers both Discord and Telegram. A single OpenAI-driven agent holds context, long-term memory, and missions while the transports only forward messages and commands.

## Highlights

- **One persona, many rooms** – Discord and Telegram reuse the same assistant instance and memory.
- **Prompt-driven behavior** – All tone, mission handling, and conversational rules live in the system prompt so replies feel adaptive instead of scripted.
- **Per-user long-term memory** – Preferences, facts, personality notes, and mission history are stored in SQLite with JSON mirrors in `mem/`.
- **Context recall** – Each reply considers recent channel history, relevant memories, and mission context before calling the model.
- **Mission privacy** – Objectives stay hidden from anyone except the creator while the agent still guides targets toward completion.
- **Selective group awareness** – Discord group replies trigger on summons, implicit questions, or recent-thread follow-ups while idle chatter is ignored.
- **Inbox archiving** – First-contact private messages are archived in `mem/inbox/` before the assistant replies.
- **GPT-5 upgrades + wallet** – The agent can upsell complex jobs to GPT-5, quote costs, accept payment, and track an on-chain wallet for payouts or fast opportunistic sends.

## Requirements

- Python 3.10+
- Discord bot token
- Telegram bot token
- OpenAI API key with Chat Completions access
- Optional: EVM RPC endpoint plus wallet address/private key if you want live transfers

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
   - Optional overrides: `MODEL` (defaults to `gpt-4.1`), `MEM_DIR`, `DISCORD_GUILD_ID`, `EVM_RPC_URL`, `EVM_WALLET_ADDRESS`, `EVM_WALLET_PRIVATE_KEY`
3. Ensure the `mem/` directory remains writable (it is created automatically on first run).

## Running

Start both transports with a single command:

```bash
python main.py
```

Discord exposes `/assignagenda <user> <goal> [timeout_hours]` and `/stopagenda <mission_id>` slash commands (optionally scoped by `DISCORD_GUILD_ID`). Mission assignment requires the creator to have an established alias or name on record so the assistant can confirm identity; the bot now records display names automatically to keep trusted contacts recognized.
Telegram offers `/assignagenda <goal> [timeout_hours]` and `/stopagenda <mission_id>` in DMs only. Each command updates the shared assistant memory so the LLM can pursue missions conversationally.

When a mission times out, the assistant automatically notifies the creator via DM with a concise summary. Mission creators can also request snapshots programmatically through `Assistant.get_mission_status("platform:user")`.

After solving a demanding task the assistant may pitch a GPT-5 deep dive. It estimates OpenAI usage, quotes either $1 or ten times the cost (whichever is larger than $1), waits for confirmation, and then runs the upgraded job. Wallet credentials let the agent advertise a fixed deposit address, check balances silently, and—when the model flags a short-term gain—emit wallet directives that the runtime executes to send ETH.

## Data Layout

- `memory.db` – SQLite store for user profiles, preferences, facts, and mission metadata.
- `mem/` – JSON mirrors and inbox archives for easier inspection.
- `mem/inbox/` – Raw text snapshots of first-contact DMs from new users.

Back up or delete these files to reset the assistant's recollections.
