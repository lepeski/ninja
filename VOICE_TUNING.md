# Voice Tuning Guide

This project keeps the live persona outside the Python core so operators can tweak the voice without touching code. Start here:

## Persona config (primary entry point)
- Location: `core/persona_config.yaml` (or drop an override in `mem/persona.yaml`).
- What it does: defines tone, privacy, lore, and nickname rules that `_build_persona_prompt` loads at runtime.
- How to modify: edit the YAML strings directly. The assistant hot-reloads changes and journals major pivots, so you can iterate without redeploying or editing Python.

## Core persona string (advanced)
- Location: `core/assistant.py`, [`Assistant._build_persona_prompt`](core/assistant.py#L1375).
- What it does: merges the YAML config with extra guardrails (alias handling, upgrade terminology, directive handling).
- How to modify: only if you need structural changes beyond what YAML can express. Keep additions terse to save tokens.

## Runtime prompt composer
- Location: `core/assistant.py`, [`Assistant._compose_system_prompt`](core/assistant.py#L2347).
- What it does: merges persona, mission context, nicknames, wallet rules, and memory blurbs before each OpenAI call.
- How to modify: adjust or append to the `prompt_parts` list. This is where you can inject extra rules (e.g., change how nicknames are surfaced or add new lore beats) or remove guidance you no longer want. Keep new lines concise—each addition increases per-message token usage.

## Placeholder / alias behavior
- Location: `core/assistant.py`, [`MemoryStore.display_name`](core/assistant.py#L1015) and [`MemoryStore.alias_for`](core/assistant.py#L1027).
- What it does: controls how unknown users map to placeholders and when the fixed acronym appears in the prompt.
- How to modify: adjust the fallback logic or alias token construction to change how anonymous users are referenced. Update the prompt composer if you introduce new placeholder semantics so the model knows what to do with them.

## Lore pacing and reveal depth
- Location: `core/assistant.py`, the lore instructions injected inside `_compose_system_prompt` and the journal/nickname helpers nearby.
- What it does: enforces how much of the Cyberhood backstory leaks out when users pry.
- How to modify: tune the strings added near the `lore protocol` line or extend the helper logic that feeds contextual cues. Keep the staged reveal structure if you still want progressive disclosure.

## Testing changes safely
1. Run `python -m compileall core transports main.py` to catch syntax issues quickly.
2. Use a private Discord or Telegram channel to validate tone changes without disrupting live missions.
3. Watch the logs for `prompt_parts` warnings—if you remove mandatory rules the assistant might start leaking mission intel.

Keeping all persona edits inside these functions ensures the rest of the mission, memory, and wallet systems continue to operate without extra rewiring.
