# AI Analytics Backend

Python backend for a desktop AI analytics/voice agent. This backend is
**UI-independent** — it exposes a FastAPI API that a future Tauri + React
desktop application (or the CLI test client, added in a later phase) will
call. No Tauri, Rust, React, Streamlit, or desktop packaging code lives here.

Being built incrementally, phase by phase. **This is Phase 1.**

## Phase 1 scope

- Project skeleton matching the target structure (empty packages for later
  phases so imports won't need restructuring as they fill in)
- `app/config.py` — typed settings, loaded from `.env` / environment
- `app/core/credentials.py` — credential abstraction (`CredentialStore`)
  with `env` and encrypted `local_file` backends for local dev, plus
  documented stub extension points for Windows Credential Manager / macOS
  Keychain (owned by the future Tauri shell)
- `app/sources/models.py` — `ProviderType`, `DataSourceConfig` (public, no
  secrets), `DataSourceCreate`/`DataSourceUpdate` (secret-carrying input
  only), `UserProfile`
- `app/sources/repository.py` — JSON-backed CRUD for sources + user
  profile; routes PATs into the `CredentialStore` and guarantees they never
  land in the persisted config file
- `app/main.py` — FastAPI app with `/health`, `/me`, `/sources` (read-only
  introspection endpoints; `/chat`, `/dashboard`, `/report` etc. are Phase 11)
- Test suite covering config, credentials, models, and the repository

## Not in this phase

Groq, LangGraph, MCP connections, source selection, SQL generation/safety,
dashboards, reports, summarization, and the full API surface. See "Roadmap"
below — each is a separate phase.

## Project structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app (Phase 1: health/me/sources only)
│   ├── config.py            # Settings (all env vars declared up front)
│   ├── dependencies.py      # Singleton wiring for repositories
│   ├── core/
│   │   ├── credentials.py   # CredentialStore abstraction + backends
│   │   └── logging.py       # Logger + secret-redaction helper
│   ├── sources/
│   │   ├── models.py        # DataSourceConfig, UserProfile, etc.
│   │   └── repository.py    # JSON persistence + credential routing
│   ├── api/                 # (empty — Phase 11)
│   ├── agent/                # (empty — Phase 2+)
│   │   ├── nodes/
│   │   └── prompts/
│   ├── llm/                 # (empty — Phase 2, Groq client)
│   ├── mcp/                 # (empty — Phase 4+)
│   ├── memory/               # (empty — Phase 3)
│   ├── analytics/            # (empty — Phase 7)
│   ├── dashboard/            # (empty — Phase 9)
│   └── reports/              # (empty — Phase 10)
├── data/                     # local JSON config + encrypted credential file (gitignored)
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

One deliberate deviation from the originally sketched structure: `app/mcp/`
currently has no `manager.py`/`connection.py` yet (added in Phase 4) — kept
as an empty package now so nothing needs to move later.

## How to run

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # or your preferred env tool
pip install -e ".[dev]"

cp .env.example .env
# edit .env: set DEFAULT_USER_DISPLAY_NAME, etc.

python -m app.main
# or: uvicorn app.main:app --reload
```

Then:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/me
curl http://127.0.0.1:8000/sources
```

## How to test

```bash
pytest -q
```

No real Groq/MCP credentials are required — this phase has no external
calls. 25 tests, all passing, covering:

- Settings defaults and env overrides
- `EnvCredentialStore` and `LocalFileCredentialStore` (including a check
  that the encrypted-at-rest file never contains the plaintext secret)
- Model invariants: `DataSourceConfig` never serializes a `pat`; two
  same-provider sources (e.g. "Sales Snowflake" / "Finance Snowflake") get
  distinct, non-name-derived IDs
- `SourceRepository`: create/get/update/delete, credential rotation,
  multi-provider configuration, PAT never written to `sources.json`
- `UserProfileRepository`: default profile creation, add/remove source

## Security notes (Phase 1)

- `DataSourceConfig` — the object that gets persisted, returned from the
  API, and will later be put into LangGraph state / prompts — has no field
  that can hold a secret. It only has `credential_ref`.
- PATs enter the system only via `DataSourceCreate.pat` /
  `DataSourceUpdate.pat`, are written straight to the `CredentialStore`, and
  are not retained by the repository afterward.
- `SourceRepository.resolve_credential()` is the only method that returns a
  raw secret. It's intended to be called immediately before use (inside the
  MCP connection layer, added in Phase 4) — callers must not log, prompt,
  or persist the result.
- The default `local_file` credential backend uses Fernet symmetric
  encryption and is explicitly documented as **dev-only**, not a substitute
  for a real OS credential store. Production/desktop use is expected to go
  through the Tauri-owned Windows Credential Manager / macOS Keychain
  stubs in `app/core/credentials.py`.

## Roadmap (per implementation plan)

1. ✅ Project setup + configuration + user/source models
2. Groq client + basic LangGraph conversational agent
3. Conversation persistence + summarization
4. MCP manager + one generic MCP connection
5. Multiple MCP source configuration + source selection node
6. MCP tool calling
7. SQL generation + safety validation
8. Multi-source analysis
9. Dashboard specification
10. Report specification + PDF generation
11. FastAPI API (`/chat`, `/dashboard`, `/report`, full `/sources` CRUD) + CLI test client
