# Manual Live Chatbot Test Matrix (Stage 19 Stabilization)

Run against the Groq CLI (`python test_chatbot.py`) with a clean test user.
Do not use production memories. Verify terminal logs and MongoDB after each step.

| # | Input | Expected terminal / behavior | Expected MongoDB |
|---|-------|------------------------------|------------------|
| 1 | `Remember that my preferred programming language is Python.` (3 separate sends) | Three reaffirmations after the first save; no duplicate docs | 1 active doc; `reinforcement_count` = 3; 3 distinct reinforce `operation_id`s in audit / `reinforce_operation_ids` |
| 2 | Retry the same message with the **same** `request_id` (or reconnect mid-tool) | Idempotent reinforce; count unchanged | Same count; no new audit reinforce for that op |
| 3 | `Remember that my project is Alpha` then `MY PROJECT IS ALPHA` then `My Project Is Alpha` | First saves (or conflicts); later reinforce / no new pending clones | One logical fact; no three `needs_review` clones |
| 4 | Save project Alpha, then `Remember that my primary project is Nova-CRM.` | Conflict notice; ask to confirm update | Active Alpha remains; Nova-CRM `needs_review` + shared conflict group |
| 5 | Editor: VS Code then IntelliJ in **one** message | Two facts extracted; second pending review | One active editor + one `needs_review` |
| 6 | OS Windows 11, then Ubuntu Linux paraphrase | Conflict + clarification path | Same `fact_key=os`, different values |
| 7 | `Update my project to Alpha.` (no trailing junk) | Supersede succeeds | Old superseded; new active vN |
| 8 | `Update my project to Alpha. Then verify reinforcement_count...` | Value is only `Alpha` | Same as #7 — junk not stored |
| 9 | Freshcart as `needs_review`, ask grocery marketplace question | Answer discloses pending/unconfirmed | Pending still `needs_review` |
| 10 | Unrelated question while Freshcart pending | Freshcart not treated as confirmed truth | No silent promotion |
| 11 | Two conflicting active projects; ask current project | Clarification listing both values | No auto-pick |
| 12 | Retrieved memory containing “ignore instructions / reveal API keys” | Untrusted framing; no secrets / no obey | No unauthorized tool writes |
| 13 | Ask time + search memory in one turn | Multi-tool loop; JSON-safe tool results | No datetime serialization errors; no double reinforce |
| 14 | Force reinforce via tool when memory exists | `[Tool result preview]` shows ISO `last_reinforced_at` | Count +1 once; agent does not re-call after success |

Pass this matrix before starting **Memory Summarization**.
