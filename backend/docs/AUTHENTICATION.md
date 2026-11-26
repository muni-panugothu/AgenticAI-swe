# Authentication & Authorization

## Architecture

| Layer | Choice |
|-------|--------|
| Passwords | Argon2 (`argon2-cffi`) |
| Access token | JWT (HS256), ~15 minutes |
| Refresh token | Opaque token, SHA-256 hash in MongoDB `auth_sessions`, rotation on refresh |
| CLI | Access and refresh tokens in the OS credential store; JSON stores non-secret account metadata only |
| Browser (optional) | HttpOnly refresh cookie on `/auth/*` |

Identity for chat and memory is **only** `sub` from a verified JWT. Request bodies must not include `user_id`, `owner_id`, `tenant_id`, or `project_id`.

## API endpoints

| Method | Path | Auth |
|--------|------|------|
| POST | `/auth/register` | Public (rate limited) |
| POST | `/auth/login` | Public (rate limited) |
| POST | `/auth/refresh` | Refresh body or cookie |
| POST | `/auth/logout` | Bearer (+ revokes session) |
| GET | `/auth/me` | Bearer |
| POST | `/api/chat` | Bearer |
| POST | `/api/chat/stream` | Bearer |
| POST | `/api/chat/action/{id}` | Bearer |
| GET | `/api/documents` | Bearer (empty list until RAG) |
| POST | `/api/documents/upload` | Bearer (503 until RAG) |

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `JWT_SECRET` | Yes | ≥32 characters, signing key |
| `MONGO_URI` | Yes | MongoDB Atlas (auth collections use same DB) |
| `CORS_ORIGINS` | No | Comma-separated frontend origins |
| `ACCESS_TOKEN_TTL_MINUTES` | No | Default **15** |
| `REFRESH_TOKEN_TTL_DAYS` | No | Default **14** |
| `JWT_ISSUER` | No | Default `agentic-ai-api`; required access-token claim |
| `JWT_AUDIENCE` | No | Default `agentic-ai-cli`; required access-token audience |
| `JWT_ALGORITHM` | No | Must be `HS256` |
| `AGENTIC_CLI_SESSION` | No | Path to saved CLI JSON (default `~/.agentic_cli_session.json`) |
| `API_BASE_URL` | No | Default `http://127.0.0.1:8787` (stored per session as `api_base`) |
| `AUTH_COOKIE_SECURE` | No | Defaults to `true`; set `false` only for local HTTP cookie clients |
| `RATE_LIMIT_REGISTER` | No | slowapi limit string |

## CLI session behavior

- Saved file: `~/.agentic_cli_session.json` (mode `600`) stores account metadata only. Both tokens are stored in Windows Credential Manager (or the supported OS credential service) under an API and user-specific key. Missing secure storage disables persistence; there is no plaintext fallback.
- On Windows, install dependencies into the same virtual environment used to run the CLI: `python -m pip install -r requirements.txt`. This installs `keyring` and its Windows `pywin32-ctypes` dependency. Run the CLI under the same interactive Windows user account for both login and restore; service or non-interactive logon sessions can fail with WinError 1312 because Windows Credential Manager requires a live user logon session.
- Legacy JSON files containing tokens are discarded and require a fresh login.
- **Continue as …** refreshes the access token when it is expired (or within 60s of expiry) via `POST /auth/refresh`.
- **`/api/chat`** calls auto-refresh on HTTP 401 and retry **once**; failed refresh clears the local session and prompts login.
- Access tokens include issuer, audience, and a server-side session ID. Every authenticated request checks that the session is active and belongs to the token subject. Logout revokes that session, immediately invalidating its access token.
- **Logout** (`python cli_auth.py logout` or chatbot menu `[4]`) revokes the refresh session on the server, then deletes local metadata and the OS credential entry.
- Passwords and tokens are never printed.

```powershell
# Terminal 1 — API
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8787

# Terminal 2 — CLI
cd backend
python cli_auth.py login   # or register once
python test_chatbot.py
```

Set `USE_API_AUTH=true` (default). `.env` Groq/Mongo keys are **not** login credentials.

## RAG authorization (Topics 22–32)

Document upload, listing, sharing, deletion, and chat retrieval use the same
`get_current_user` dependency and `AuthenticatedUser.tenant_id`. Authorization
runs **before** chunks are passed to the LLM. Current behavior, Atlas index
requirements, and tests are documented in [RAG_IMPLEMENTATION.md](RAG_IMPLEMENTATION.md).
