# Agentic Engineer — Web App

Professional React web client for the existing FastAPI Agentic AI Software Engineer backend. The Expo Android app in `../app` continues to use the same API.

## Stack

- React + Vite + TypeScript
- Tailwind CSS v4
- React Router
- Lucide icons
- `react-markdown` + syntax highlighting

## Setup

```bash
cd web
cp .env.example .env
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

Backend (separate terminal):

```bash
cd backend
# ensure CORS allows http://localhost:5173 (defaults now include it)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8787
```

Environment:

```env
VITE_API_BASE_URL=http://localhost:8787
```

If you open the site from another device, set `VITE_API_BASE_URL` to the host LAN IP and add that origin to backend `CORS_ORIGINS`.

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Dev server on port 5173 |
| `npm run build` | Typecheck + production build |
| `npm run preview` | Preview production build |
| `npm test` | Unit tests (Vitest) |

## Connected backend endpoints

- `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh`, `POST /auth/logout`, `GET /auth/me`
- `POST /api/chat`, `POST /api/chat/stream`
- `GET /api/documents`, `POST /api/documents/upload`, `DELETE /api/documents/{id}`
- `GET /health`

## Known limitations

- **No server-side conversation history API** — chat messages are ephemeral browser session storage scoped to the signed-in user ID.
- **Streaming** is the backend’s SSE envelope (`meta` → `token` → `done`). The server currently builds the full reply then chunks it; the client still parses real SSE (not timer-simulated).
- **Access tokens** live in `sessionStorage` for Bearer API compatibility. Refresh may also use the backend HttpOnly cookie (`credentials: 'include'`). sessionStorage is **not** equivalent to HttpOnly cookie protection.
- Agent tool execution details are **not** exposed as structured SSE events by the backend; only the final assistant text (and RAG citations when present) is shown.

## Security notes

- Passwords are never stored in the frontend.
- Logout clears tokens and per-user chat UI state.
- Document access and chat authorization remain enforced by the backend JWT identity.
