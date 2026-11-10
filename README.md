# Agentic AI Software Engineer — Practice

Advanced memory intelligence for an Agentic AI Software Engineer chatbot.

## Architecture

```
Agentic AI Software Engineer Chatbot
  (Groq LLM · Agent loop · Authenticated user context)
                 │
                 ▼
Memory Intelligence Layer
  Fact extraction · Conflict engine · Version manager
  Lifecycle engine · Retrieval engine · Audit engine
                 │
                 ▼
MongoDB Atlas
  Memories · Versions · Conflict records · Audit events
```

## Stages

- **Stage 17 — Conflict Intelligence:** semantic fact extraction, contradiction confidence scoring, versioned supersede transactions, explicit confirmation, audit events.
- **Stage 18 — Lifecycle:** freshness/importance ranking, reinforcement + access tracking, stale identification (no auto-delete), Mongo persistence tests.

## LLM memory tools

`search_memory`, `save_memory`, `detect_memory_conflict`, `request_memory_update`, `confirm_memory_update`, `reinforce_memory`, `forget_memory`

`user_id` is injected by the orchestrator — never accepted from the model.

## Quick tests

```bash
cd backend
python -m unittest discover -s tests -p "test_*.py" -v
python tests/test_stage17_mongodb_transaction_fixed.py
python tests/test_stage18_mongodb_lifecycle.py
```
