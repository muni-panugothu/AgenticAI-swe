# Authentication & Authorization

## Architecture

| Layer | Choice |
|-------|--------|
| Passwords | Argon2 (`argon2-cffi`) |
| Access token | JWT (HS256), ~15 minutes |
| Refresh token | Opaque token, SHA-256 hash in MongoDB `auth_sessions`, rotation on refresh |
| Mobile / CLI | Bearer access token + refresh in SecureStore / `~/.agentic_cli_session.json` |
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
| `ACCESS_TOKEN_TTL_MINUTES` | No | Default 15 |
| `REFRESH_TOKEN_TTL_DAYS` | No | Default 14 |
| `COOKIE_SECURE` | No | `true` in production HTTPS |
| `RATE_LIMIT_REGISTER` | No | slowapi limit string |

## CLI

```powershell
cd backend
python cli_auth.py login
$env:USE_API_AUTH='true'
python test_chatbot.py
```

## RAG readiness (Topics 22–32)

The same `get_current_user` dependency and `AuthenticatedUser.tenant_id` should guard document upload, chunk storage, and vector query filters. Authorization must run **before** chunks are passed to the LLM. No RAG collections are created in the auth milestone.
