# data-saha — Backend

**The Python backend powering data-saha**, a desktop AI analytics agent that lets you ask natural-language questions about your data warehouses (Snowflake, Redshift, BigQuery, Databricks, Fabric, Supabase, and any other MCP-compatible source) and get real, grounded answers — not invented numbers — plus dashboards and PDF reports generated on demand.

This repository is **the backend only**. It is a complete, independently runnable, UI-independent FastAPI service. The desktop UI (Tauri + React) lives in a separate `frontend/` folder and talks to this backend exclusively over HTTP — this backend has no knowledge of, and no dependency on, any frontend technology.

> **This is not a toy backend.** Every piece described below — MCP connections, SQL safety validation, multi-source data normalization, dashboard/report generation, the full API — has been built, tested, and verified against real local MCP servers and a real running HTTP server, not just unit-tested in isolation. 163+ automated tests back this codebase.

---

## Table of contents

- [What this backend actually does](#what-this-backend-actually-does)
- [Why this architecture](#why-this-architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Configuration reference](#configuration-reference)
- [Running the server](#running-the-server)
- [API reference](#api-reference)
- [The agent pipeline, end to end](#the-agent-pipeline-end-to-end)
- [Security model](#security-model)
- [Data source management](#data-source-management)
- [Dev tools / CLI scripts](#dev-tools--cli-scripts)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [What this backend deliberately does NOT do](#what-this-backend-deliberately-does-not-do)

---

## What this backend actually does

data-saha's backend is an **agentic analytics engine**, not a thin API wrapper. A single question like *"How much revenue did we make in Q4?"* triggers a real pipeline:

1. **Decides which configured data source(s) are relevant** to the question — correctly distinguishing between, say, a "Sales Snowflake" and a "Finance Snowflake" instance of the same provider, using their descriptions, not just their names.
2. **Connects to the right MCP server(s) only** — never eagerly connects to every configured source on every message.
3. **Discovers each source's tools dynamically** — never assumes a source exposes `execute_sql` or any specific tool; whatever the MCP server actually reports is the source of truth.
4. **Runs a real agentic tool-calling loop** — the model can call a tool, read the result, decide it needs another tool, call that too, and only then answer. This is not a single-shot "ask → query → answer" pipeline.
5. **Validates every SQL statement before it ever reaches a database** — a real parser-based safety layer that blocks writes, admin operations, multi-statement injection, and comment-based obfuscation tricks, regardless of what the model intended.
6. **Normalizes results from different sources into a common shape** and, when a question spans more than one source, computes actual cross-source comparisons in pandas before the model ever sees the data — rather than hoping the model does the arithmetic correctly from two separate blobs of text.
7. **Remembers the conversation** across restarts, with automatic summarization once it grows long, so follow-up questions ("and how many of those were from India?") resolve correctly against what was just discussed.
8. **Can turn any of this into a structured dashboard specification or a downloadable PDF report** on request — without ever generating a line of frontend code itself.

Everything above is reachable over a clean FastAPI HTTP interface designed for a desktop app to consume.

---

## Why this architecture

A few decisions in this codebase are deliberate and worth understanding before you extend it:

- **Credentials are never stored in plain text and never reach the LLM.** Every data source's access token (PAT) is routed through a pluggable `CredentialStore` the moment it's provided, and the rest of the application only ever handles a non-secret `credential_ref`. The token is resolved back to its real value only at the exact moment a connection needs it, and it never appears in logs, API responses, LangGraph state, or LLM prompts.
- **SQL safety is enforced in code, not by asking the model nicely.** A dedicated validator parses every SQL string before execution: strips comments (so a dangerous statement can't be hidden inside `/* ... */` or after `--`), requires the statement to start with `SELECT`/`WITH`, and separately scans the whole string for write/admin keywords and known data-exfiltration patterns — anywhere in it, not just at the start.
- **The agent loop is a real loop, not a wrapper.** Tool calling uses the model's native function-calling protocol with a bounded iteration limit, full conversation context, and graceful degradation (a failed tool call is reported back to the model, not silently swallowed).
- **One canonical data-gathering pipeline, reused everywhere.** Chat, dashboard generation, and report generation all go through the same source-selection → MCP discovery → tool-calling → SQL-validation → normalization pipeline, so a warehouse is never queried twice for logically the same request, and a fix in one place fixes all three.
- **Everything downstream of a tool call works with normalized, structured data — not raw text.** Tool results are parsed into pandas DataFrames when tabular, which is what makes cross-source comparison, dashboard charts, and report tables possible without re-querying or re-parsing.

---

## Tech stack

| Concern | Library |
|---|---|
| Web framework | FastAPI (async, auto-generated OpenAPI docs at `/docs`) |
| Agent orchestration | LangGraph |
| LLM | Groq (OpenAI-compatible function calling) |
| MCP client | Official `mcp` Python SDK (Streamable HTTP transport) |
| Data validation / models | Pydantic v2 |
| Data analysis | pandas |
| PDF generation | ReportLab (native chart rendering, no external image dependency) |
| Config | pydantic-settings (`.env`-driven) |
| Testing | pytest, pytest-asyncio, FastAPI `TestClient`, `httpx` |

Python 3.11+.

---

## Project structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app, CORS, lifespan startup
│   ├── config.py                # All settings, env-driven
│   ├── dependencies.py          # Singleton wiring (repositories, manager, graph)
│   │
│   ├── api/                     # Thin HTTP adapters — no business logic
│   │   ├── routes_chat.py
│   │   ├── routes_sources.py
│   │   ├── routes_dashboard.py
│   │   ├── routes_report.py
│   │   └── routes_conversation.py
│   │
│   ├── agent/
│   │   ├── graph.py              # LangGraph: load_context → source_selector →
│   │   │                         #   (tool_executor | conversation) → summarizer
│   │   ├── state.py               # AgentState
│   │   ├── runner.py              # run_turn(): persistence ↔ graph bridge
│   │   └── nodes/
│   │       ├── context.py         # injects user profile into state
│   │       ├── source_selector.py # decides which MCP source(s) are relevant
│   │       ├── tool_executor.py   # the real agentic tool-calling loop
│   │       ├── conversation.py    # plain-chat fallback (no source needed)
│   │       └── summarizer.py      # trims + summarizes long conversations
│   │
│   ├── llm/
│   │   └── groq_client.py         # thin Groq wrapper, model never hard-coded
│   │
│   ├── mcp/
│   │   ├── manager.py             # lazy, per-source connection pool
│   │   ├── connection.py          # one MCP session's full lifecycle
│   │   └── models.py              # ToolInfo, ToolCallResult, NormalizedToolResult
│   │
│   ├── sources/
│   │   ├── models.py               # DataSourceConfig (public) vs DataSourceCreate (secret-carrying input)
│   │   └── repository.py           # CRUD + credential routing
│   │
│   ├── core/
│   │   ├── credentials.py          # CredentialStore abstraction (env / local_file / OS stubs)
│   │   └── logging.py
│   │
│   ├── memory/
│   │   ├── conversation.py         # disk-backed ConversationStore
│   │   └── summarizer.py           # token-threshold trigger + Groq-backed summarization
│   │
│   ├── analytics/
│   │   ├── sql_validator.py        # real SQL safety layer
│   │   ├── dataframe.py            # tool-result → pandas normalization + cross-source combine
│   │   └── gathering.py            # shared data-gathering pipeline (chat/dashboard/report)
│   │
│   ├── dashboard/
│   │   └── specification.py        # generate_dashboard() — structured spec + datasets
│   │
│   └── reports/
│       ├── specification.py        # generate_report() — spec with deterministic + LLM fields
│       └── generator.py            # ReportLab PDF rendering
│
├── scripts/                        # Dev-only CLIs (see below)
├── tests/                          # 163+ tests, several hitting real local MCP servers
├── data/                           # Local persistence (gitignored: sources, conversations, credentials)
├── .env.example
└── pyproject.toml
```

---

## Quickstart

```bash
cd backend
python -m pip install --upgrade pip     # pip ≥ 21.3 required for editable installs
pip install -e ".[dev]"

cp .env.example .env
# edit .env — see Configuration reference below
```

Minimum viable `.env` to get a working agent:

```bash
GROQ_API_KEY=your-real-groq-key
GROQ_MODEL=openai/gpt-oss-20b          # verify current model names at console.groq.com — Groq's lineup changes
CREDENTIAL_STORE_BACKEND=local_file
LOCAL_CREDENTIAL_STORE_KEY=            # generate with the command below
DEFAULT_USER_DISPLAY_NAME=Mike
```

Generate a local credential encryption key once:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

> ⚠️ **Do not change `LOCAL_CREDENTIAL_STORE_KEY` after sources are registered.** It's the encryption key for every stored access token. Changing it makes previously stored credentials permanently unreadable.

Register your first data source:

```bash
python -m scripts.manage_sources add \
  --display-name "Supabase Prod" \
  --provider generic_mcp \
  --mcp-url "https://mcp.supabase.com/mcp?project_ref=YOUR_PROJECT_REF" \
  --pat "your-real-pat" \
  --description "Enterprise e-commerce PostgreSQL database — customers, orders, inventory, finance" \
  --business-domain "e-commerce"
```

> `description` and `business_domain` aren't decoration — the agent uses them to decide *which* source answers a given question, especially when you have more than one. Write real ones.

Run the server:

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive API docs, or talk to it from the terminal:

```bash
python -m scripts.chat_cli
```

---

## Configuration reference

All settings are read from `.env` (or real environment variables) via `app/config.py`. Nothing here is hard-coded.

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `development` \| `test` \| `production` |
| `APP_DATA_DIR` | `./data` | Local persistence root (sources, conversations, credentials) |
| `CREDENTIAL_STORE_BACKEND` | `env` | `env` (in-process only — see warning below) or `local_file` (encrypted, persistent) |
| `LOCAL_CREDENTIAL_STORE_KEY` | — | Fernet key, required when backend is `local_file` |
| `DEFAULT_USER_ID` | `local-user` | Single local user's id (this is a single-user desktop app) |
| `DEFAULT_USER_DISPLAY_NAME` | `User` | Used in greetings and prompts |
| `DEFAULT_USER_TIMEZONE` | `UTC` | — |
| `DEFAULT_USER_LANGUAGE` | `en` | — |
| `GROQ_API_KEY` | — | Required for any LLM-backed route |
| `GROQ_MODEL` | — | Required; never hard-coded. Groq's model lineup changes — verify current names |
| `SUMMARY_TRIGGER_TOKENS` | `6000` | Approximate token threshold that triggers conversation summarization |
| `RECENT_MESSAGES_KEEP` | `10` | Messages kept verbatim after a summarization trigger |
| `SQL_SAFETY_MODE` | `strict` | `strict` or `permissive` (permissive only relaxes the supplementary unsafe-function blocklist — writes/DDL/multi-statement are **always** blocked in both modes) |
| `CORS_ORIGINS` | Vite/Tauri defaults | Comma-separated allow-list for a Tauri/React or Vite-dev frontend |

> ⚠️ **`CREDENTIAL_STORE_BACKEND=env` only persists a credential for the lifetime of the process that set it.** If you register a source with `manage_sources add` and then probe it from a *separate* process (another CLI run, the server, etc.), the token will not be found. Use `local_file` for anything beyond a single, single-process test.

---

## Running the server

```bash
# dev, auto-reload
uvicorn app.main:app --reload

# production-ish
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On startup the app validates config, touches the source repository (so a bad credential backend fails fast at boot, not on first request), and logs the active user/source count.

---

## API reference

Interactive docs are always available at `/docs` (Swagger) and `/redoc` once the server is running — treat this table as a summary, not the source of truth.

### `GET /health`
```json
{"status": "ok"}
```

### `GET /me`
```json
{"user_id": "local-user", "display_name": "Mike", "timezone": "Asia/Kolkata", "preferred_language": "en", "configured_data_sources": ["generic_mcp-db6eb241"]}
```

### `POST /chat`
The main conversational endpoint. Runs the full agent pipeline described below.

**Request**
```json
{"session_id": "local-user-main", "message": "How much revenue did we make in Q4?"}
```
`session_id` is optional — omit it to use the single default session.

**Response**
```json
{
  "session_id": "local-user-main",
  "message": "Q4 revenue was $1,200,000, led by the US region.",
  "dashboard_available": true,
  "report_available": true,
  "sources_used": ["generic_mcp-db6eb241"]
}
```
`dashboard_available`/`report_available` are only `true` when the answer actually drew on live source data — they're a signal for the UI to enable those buttons for this specific answer.

**Errors:** `502` with `{"detail": "..."}` when the LLM is unavailable/misconfigured.

### `GET /conversation/{session_id}`
Returns the persisted conversation record — recent messages, rolling summary, active source ids. Unknown session ids return an empty record, not a 404.

### `GET /sources`
Returns every configured source. **Never contains a credential** — the response model has no field capable of holding one.

### `POST /sources` → `201`
**Request**
```json
{
  "display_name": "Sales Snowflake",
  "provider": "snowflake",
  "mcp_url": "https://...",
  "pat": "write-only, never echoed back",
  "description": "Contains CRM opportunities and sales activity",
  "business_domain": "sales",
  "enabled": true
}
```
`provider` ∈ `snowflake | redshift | bigquery | databricks | fabric | generic_mcp` (use `generic_mcp` for Supabase or anything else not explicitly listed).

### `DELETE /sources/{source_id}` → `204`
`404` if the id doesn't exist.

### `GET /sources/{source_id}/health`
Always `200`:
```json
{"source_id": "...", "healthy": false, "message": "No stored credential for source '...'. ..."}
```
Connects (if needed), checks, then disconnects — never leaves a lingering connection open.

### `POST /dashboard`
**Request:** `{"question": "Show me sales revenue by region"}`

**Response:** a `DashboardSpecification` (title, description, `charts[]`, `metrics[]`, `filters[]`) plus `datasets[]` — the actual normalized rows each chart needs. The backend **never generates frontend code**; this is pure data + spec for a UI to render however it likes.

### `POST /report`
**Request:** `{"question": "...", "format": "json" | "pdf"}` (default `json`)

- `format: "json"` → a `ReportSpecification` (title, summary, key_findings, metrics, tables, charts, data_sources, query_information, generation_timestamp) + `datasets[]`. Facts (`data_sources`, `query_information`, `tables`, numeric `metrics`) are computed deterministically from the real gathered data; only the narrative (`summary`, `key_findings`) and chart suggestions are LLM-generated, explicitly instructed never to invent a number or finding the data doesn't support.
- `format: "pdf"` → a real rendered PDF (`application/pdf`), including native bar/line/pie charts, streamed back with a filename.

---

## The agent pipeline, end to end

```
User message
      │
      ▼
 load_context ──── injects display name, timezone, etc. into state
      │
      ▼
 source_selector ── LLM call: which configured source(s), if any, match
      │              this question? Uses display_name/description/
      │              business_domain (never provider name alone) to
      │              disambiguate sources, AND recent conversation
      │              history to resolve follow-ups ("in that, how many
      │              were from India?") against what was just discussed.
      │
      ├── no source needed ──► conversation node (plain chat reply)
      │
      └── source(s) selected
                │
                ▼
         tool_executor ── discovers each source's REAL tools (never
                │          assumed), lets the model call them (bounded
                │          loop, may call more than one tool, more than
                │          once), validates any SQL before it reaches a
                │          database, normalizes successful results into
                │          pandas, injects a computed cross-source
                │          comparison once 2+ sources have data, then
                │          produces the final grounded answer
                │
                ▼
         summarizer ── trims + summarizes once the conversation exceeds
                        SUMMARY_TRIGGER_TOKENS, keeping the most recent
                        RECENT_MESSAGES_KEEP messages verbatim
                │
                ▼
         persisted to disk, response returned
```

`generate_dashboard()` and `generate_report()` reuse the exact same source-selection → tool-discovery → tool-calling → SQL-validation → normalization pipeline (`app/analytics/gathering.py`) rather than duplicating it — one warehouse query, reused for whichever capability needs it.

---

## Security model

- **Credentials never touch the LLM, logs, API responses, or persisted conversation/source files.** The only object that can hold a raw token is the write-only `DataSourceCreate` input model, which is consumed once by the repository and converted into a `DataSourceConfig` (no secret field exists on that type at all — it's not redaction, it's that the field doesn't exist).
- **A token is resolved to its real value exactly once per connection, immediately before use**, and is held only in memory as an HTTP `Authorization` header for the lifetime of that connection.
- **SQL safety is a real validator, not a prompt instruction.** Default-deny: only `SELECT`/`WITH` statements pass; everything else — write/DDL/admin keywords, multiple statements, comment-hidden second statements, known exfiltration function patterns — is rejected before any MCP tool is called, regardless of which source or tool is involved.
- **CORS is an explicit allow-list**, never a wildcard, since this API can trigger live warehouse queries.

---

## Data source management

Three equivalent ways to manage sources — all go through the same repository, so they're interchangeable:

1. **HTTP API** — `GET/POST/DELETE /sources`, `GET /sources/{id}/health` (what the frontend uses)
2. **CLI** — `python -m scripts.manage_sources {add|list|remove}`
3. Direct repository use in a Python shell, if scripting something custom

---

## Dev tools / CLI scripts

| Script | Purpose |
|---|---|
| `scripts/manage_sources.py` | add / list / remove data sources without the API |
| `scripts/chat_cli.py` | talk to the agent from the terminal; persists across restarts |
| `scripts/mcp_probe.py` | connect to one configured source, print its real discovered tools, optionally call one — the fastest way to debug a connection issue in isolation |

Example debugging workflow for a misbehaving source:
```bash
python -m scripts.manage_sources list                 # confirm the id
python -m scripts.mcp_probe --id <id>                  # health + tool discovery
python -m scripts.mcp_probe --id <id> --call <tool_name> --args '{"query":"select 1"}'
```

---

## Testing

```bash
pytest -q
```

163+ tests, requiring no real Groq key or real MCP credentials — every external call is mocked. A meaningful subset goes further and spins up **real local MCP servers** (using the official `mcp` SDK's server-side API) in-process to exercise the genuine connect → discover → call → disconnect lifecycle, SQL validation blocking a real attempted `DROP TABLE` before it reaches a server, and multi-source cross-comparison against two independent live servers — not just mocks of the MCP layer.

---

## Troubleshooting

**`Health: FAILED ('<some-id>')` with no other detail**
That bare-looking message is a `CredentialNotFoundError` — almost always one of:
- You're on `CREDENTIAL_STORE_BACKEND=env` and are testing from a different process than the one that registered the source → switch to `local_file`.
- `LOCAL_CREDENTIAL_STORE_KEY` changed since the source was added → re-add the source.
- You're using a stale source id from before a `remove`/re-`add` cycle → run `manage_sources list` and use the id it actually prints, not one from memory or an old terminal scrollback.

**`Failed to connect to MCP source...: Server returned an error response`**
Check `mcp_url` is the provider's actual MCP endpoint, not its database/API URL (for Supabase specifically: `https://mcp.supabase.com/mcp?project_ref=YOUR_PROJECT_REF`, not your project's `.supabase.co` URL). Also check your access token was created with the provider's required scopes.

**A follow-up question doesn't use the data source it should**
Confirm you're on a build that includes recent-conversation-history in `source_selector` (search `app/agent/nodes/source_selector.py` for `RECENT_HISTORY_WINDOW`) — earlier versions only looked at the single latest message, which breaks short follow-ups like "and how many of those were from India?".

**`pip install -e .` fails with an editable-install error**
Your `pip` is too old for PEP 660 editable installs from a `pyproject.toml`-only project. Run `python -m pip install --upgrade pip` first.

---

## What this backend deliberately does NOT do

By design, none of the following live here — they belong to the Tauri/React frontend in `frontend/`, or haven't been built yet:

- No UI, no React/Tauri code, no knowledge of either
- No microphone input or audio playback (the frontend owns text-to-speech)
- No OS keychain integration — `app/core/credentials.py` has documented stub extension points (`WindowsCredentialManagerStore`, `MacKeychainStore`) for the Tauri shell to implement
- No desktop packaging
- Server-side caching across `/dashboard`/`/report` calls — each call re-queries the source fresh, by design, since this is a single-user desktop backend talking to live warehouses

This backend is complete and ready for a frontend to be built against it.