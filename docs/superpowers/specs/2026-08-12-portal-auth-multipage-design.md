# Multi-page portal + auth — design spec

Written 2026-08-12. Scope: turn the single-page Shona voice cloner UI into
a real portal — separate pages for Generate / Voices / History behind
username+password login, with voices and generations scoped per user.

## Context

The previous milestone (spec: `2026-08-12-voice-library-scaling-design.md`)
added a SQLite-backed voice library, an async generation queue, and
generation history — all on one static page (`app/static/index.html`) with
no accounts; every voice and generation is globally visible to anyone who
can reach the app.

The user wants a "portal," not one long page. Scope, decided via
brainstorming:

1. **Auth** — simple username/password, self-signup, cookie session. No
   OAuth, no email verification, no password reset, no admin roles, no
   login rate limiting.
2. **Per-user data** — `Voice` and `Generation` rows belong to a user;
   every read/write is scoped to the logged-in user.
3. **Real pages** — Login/Register, Generate, Voices (library), History
   (dashboard) as separate static HTML files, not one scrolling page.
4. **No data migration** — existing rows in `storage/db.sqlite3` are dev
   data and are dropped; `user_id` is a required (non-nullable) FK from
   the start.
5. **No new framework** — plain static HTML + shared vanilla JS, matching
   how `index.html` was already built. No Jinja2, no build step.

Explicitly out of scope: OAuth, password reset/email, admin roles, rate
limiting, remembering the previous single-page `index.html` layout (it is
replaced, not kept as an alternate view).

## Data model changes (`app/models.py`)

New table:

```python
class User(SQLModel, table=True):
    id: str  # uuid, primary key
    username: str  # unique, required
    password_hash: str
    created_at: datetime
```

`Voice` and `Generation` each gain a required field:

```python
    user_id: str = Field(foreign_key="user.id")
```

No migration path — the app is redeployed against a fresh
`storage/db.sqlite3` (or the existing dev file is deleted), so `user_id`
can be non-nullable immediately rather than backfilled.

## Auth (`app/services/auth.py`, `app/routers/auth.py`)

- Passwords hashed with `passlib[bcrypt]` (new dependency).
- Sessions are a signed cookie via `itsdangerous` (new dependency) holding
  the user id — no server-side session table. The cookie is set
  `httponly`, `samesite=lax`; not marked `secure` by default since this
  runs over plain HTTP on a LAN/GPU box, but the flag is a one-line change
  if it's ever put behind TLS.
- Endpoints, all under `/api/v1/auth`:
  - `POST /register` — `{username, password}`. 400 if username taken or
    password empty. Creates the user, sets the session cookie, returns
    `{username}`.
  - `POST /login` — `{username, password}`. 401 on bad credentials. Sets
    the session cookie, returns `{username}`.
  - `POST /logout` — clears the cookie.
  - `GET /me` — returns `{username}` from the current session, or 401 if
    not logged in. Used by every page on load to decide whether to
    redirect to `/login.html`.
- FastAPI dependency `get_current_user(request) -> User`: reads the
  session cookie, verifies the signature, loads the `User` row. Raises
  `HTTPException(401)` if missing/invalid/user no longer exists. Every
  voice and generation route in `cloning.py` takes this as a dependency
  and scopes its DB query to `user.id`.

## Voice/generation endpoints — scoping changes

All existing behavior from the prior spec is preserved; every handler
additionally depends on `get_current_user` and filters/stamps by
`user_id`:

- `POST /api/v1/voices/upload` — sets `Voice.user_id = current_user.id`.
- `GET /api/v1/voices` — `WHERE user_id = current_user.id`.
- `DELETE /api/v1/voices/{id}` — 404 if the voice doesn't exist *or*
  belongs to another user (don't leak existence across accounts).
- `POST /api/v1/generations` — validates the voice belongs to the current
  user (404 otherwise), sets `Generation.user_id = current_user.id`.
- `GET /api/v1/generations/{id}`, `.../audio` — 404 if the generation
  belongs to another user.
- `GET /api/v1/generations` — `WHERE user_id = current_user.id`, plus the
  existing optional `voice_id` filter.

## Pages (`app/static/`)

- `login.html` — single page, a form that toggles between "Log in" and
  "Register" modes (no separate register.html — same fields, different
  submit target). On success, redirects to `/generate.html`.
- `generate.html` — the core synth flow: a compact voice picker (select
  from existing voices; "manage voices" links to `voices.html` rather than
  upload happening inline here), text input, language select, generate
  button, polling status, audio result. This replaces the "Step 2" card
  from the old `index.html`.
- `voices.html` — the library: upload form (file + label), list of the
  user's voices as rows/cards, each with a delete button. This absorbs the
  old "Step 1" upload card plus becomes the place voices are managed
  (no rename in this iteration — not part of the approved scope; delete +
  re-upload covers renaming for now).
- `history.html` — the user's full generation history: list, newest
  first, filterable by voice (dropdown of the user's voices), each row
  shows text, status, and a play link when done. Replaces the old "Step 3"
  card, no longer capped to 20 inline — paginate with a simple
  "load more" (offset-based) if the list exceeds one page (50 rows).
- `common.js` — shared across all four pages:
  - `esc(s)` — the existing HTML-escaping helper (carried over as-is).
  - `requireAuth()` — calls `GET /api/v1/auth/me` on page load; redirects
    to `/login.html` on 401. Called by generate/voices/history pages, not
    by login itself.
  - `renderNav(activePage)` — injects a shared nav bar (Generate / Voices
    / History / Log out) into a `<nav id="nav">` placeholder each
    authenticated page includes.
  - `apiFetch(url, opts)` — thin `fetch` wrapper that always sends
    `credentials: 'same-origin'` (so the session cookie goes along) and
    redirects to `/login.html` on any 401 response.

`app/main.py` serves each `.html` file at its own path (`/login`,
`/generate`, `/voices`, `/history`), same pattern as the existing `/`
route for `index.html`. The root `/` redirects to `/generate` (which then
client-side-redirects to `/login` if unauthenticated). `index.html` is
deleted — it's fully superseded by the four new pages.

## Config (`app/config.py`)

Add `session_secret: str` (used by `itsdangerous`), read from
`CLONER_SESSION_SECRET` with no hardcoded default in code — if unset at
startup, generate a random one at process start and log a warning that
sessions won't survive a restart (acceptable for this scope; documented,
not silently broken).

## Dependencies

Add `passlib[bcrypt]` and `itsdangerous` to `pyproject.toml` and
`requirements.txt`.

## Error handling

- Register with a taken username → 400 with a clear message.
- Register/login with empty username or password → 400.
- Login with wrong username/password → 401 (don't reveal which field was
  wrong).
- Any voice/generation endpoint without a valid session → 401.
- Any voice/generation endpoint targeting another user's resource → 404
  (not 403 — avoids confirming the resource exists under a different
  account).

## Testing

- Unit: password hashing round-trip, session cookie sign/verify, register
  duplicate-username rejection.
- Integration (extends the existing `tests/` suite and `client` fixture):
  register two users, confirm user A cannot see/generate against user B's
  voices (404s as specified above), confirm `GET /api/v1/voices` only
  returns the caller's own voices.
- Manual: register, log in, log out, log back in; navigate all four pages
  via the nav bar; confirm an unauthenticated visit to `/generate.html`
  bounces to `/login.html`.
