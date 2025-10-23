# Improvement Concepts

This document records potential enhancements surfaced during review. Each section outlines practical options without committing the running codebase to the change.

## Narrative progression for lore
- Track a `lore_state` value in SQLite keyed by user so fragments can advance in sequence when someone keeps probing the backstory.
- Introduce a lightweight state machine (e.g., `intro → cyberhood → exile → crystal hunt`) and let the assistant move forward only after the user explicitly asks for the next fragment.
- Allow the model to append brief reflections to the journal whenever a fragment is revealed, so later prompts can mention that specific beat without retelling everything.

## Relevance-pruned context
- Add a semantic similarity filter that keeps only the most relevant prior exchanges (top-k embedding search) before composing the OpenAI prompt.
- Down-rank stale mission or wallet notes by age so rarely used facts fall out of the `context_ram` buffer automatically.
- Maintain separate queues for mission, lore, and casual chatter, letting the assistant pull only the queue aligned with the current intent.

## Centralised persona rules
- Move tone, privacy, lore, and nickname guidance into a single YAML or JSON config so non-coders can adjust phrasing without editing Python files.
- Build a small loader that hot-reloads the persona config on change, keeping the runtime flexible for rapid experimentation.
- Store historical persona versions in the journal to trace when major stylistic pivots occurred.

## Missions with strategic memory
- Persist a `mission_strategy` field that summarises why the mission exists, refreshing it whenever the creator adds context.
- Teach the mission evaluator to consult the strategy summary before judging progress, ensuring follow-up questions stay on-mission.
- When missions conclude, archive a short “what worked / what failed” note so future missions for the same creator can adapt automatically.

