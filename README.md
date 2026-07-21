---
license: mit
title: Face school Attendance System
sdk: docker
emoji: 📚
colorFrom: yellow
colorTo: yellow
---

# Face Attendance System

A Flask app that takes student attendance two ways at once: it has to
**recognize your face** (a CNN embedding matched by cosine similarity) *and*
**confirm you're on campus** (GPS checked against a geofence). Either signal
alone is easy to fool. Together, someone would need to spoof their phone's
GPS *and* pass as the registered student's face, at the same moment, to fake
a check-in.

Built to run locally or as a Hugging Face Spaces Docker app.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Project structure — every file explained](#project-structure--every-file-explained)
3. [Setup — local development](#setup--local-development)
4. [Setup — Hugging Face Spaces](#setup--hugging-face-spaces)
5. [Configuration reference](#configuration-reference)
6. [Route reference](#route-reference)
7. [Feature walkthrough](#feature-walkthrough)
8. [Security model](#security-model)
9. [Data & privacy](#data--privacy)
10. [Known limitations](#known-limitations)
11. [Troubleshooting](#troubleshooting)

---

## How it works

### Face recognition

Registering a student and recognizing them later both go through the same
pipeline, in `face_engine.py`:

1. **Detect** — OpenCV's Haar Cascade looks for a face first, since it's
   fast. If it finds nothing (bad angle, poor lighting), **MTCNN** — a small
   cascaded CNN — is tried as a fallback, since it copes much better with
   harder cases at the cost of being slower.
2. **Quality check** — the cropped face is checked for size, blur
   (Laplacian variance), and brightness. Registration surfaces these as
   warnings so a bad photo doesn't quietly hurt accuracy later.
3. **Embed** — **MobileNetV2** (frozen ImageNet weights, `pooling='avg'`)
   maps the 96×96 face to a 1280-dimension vector. No training happens
   here — it's pure feature extraction. That's the whole reason registering
   a student is instant instead of needing a retrain step.
4. **Match** — a new face's embedding is compared via cosine similarity
   against every embedding stored in the gallery
   (`DATA_DIR/embeddings/embeddings.json`). A match is accepted only if it
   clears `MATCH_THRESHOLD` **and** beats the second-best candidate by at
   least `MATCH_MARGIN`. The margin check is what catches "two plausible
   but wrong" matches that a bare threshold would let slip through.

This mirrors how production face-recognition systems are actually built —
detect → embed → cosine-similarity-with-threshold — just with a lighter,
CPU-friendly model appropriate for a free-tier deployment.

### Geofencing

`geofence.py` handles the location half. An admin sets a campus center point
and radius (either via the `/settings` map or environment variables); every
check-in's GPS coordinates are compared against that center using the
Haversine formula (great-circle distance between two lat/lng points).

Two modes, chosen on `/settings`:

- **Strict** — check-ins outside the radius are blocked outright.
- **Flag** — check-ins outside the radius are still accepted (face
  recognition still has to pass), but the record is marked
  `location_verified = False` for later review on the dashboard. Useful
  while you're dialing in a radius, since indoor GPS can drift 50–100m even
  for someone genuinely standing on campus.

**Be honest with yourself about what this does and doesn't prove.** Browser
GPS is not tamper-proof — there's no "mock location" detection API exposed
to web pages the way native Android/iOS apps have. A determined person can
override their reported coordinates via dev tools or a spoofing app. What
this feature actually raises the bar on is casual "I'll just mark myself
present from home" behavior, and — combined with face recognition — turns a
single easy spoof into needing two simultaneous ones. Don't oversell it as
airtight; the README says this directly so nobody downstream assumes it's
cryptographic proof of presence.

### Why embeddings instead of a trained classifier

An earlier version of this project trained a fresh softmax classifier every
time a new student was added — which meant needing at least 2 students
before anything worked, retraining on everyone whenever one more was added,
and a classifier that always confidently picks *some* class even for a
total stranger's face (softmax doesn't have a "none of the above" option).
Switching to embeddings + similarity matching fixes all three: one student
is enough, adding another is instant, and a stranger is rejected by falling
below the threshold rather than getting mis-classified as somebody's photo.

---

## Project structure — every file explained

```
├── app.py                  # Flask routes — the request/response layer
├── face_engine.py           # detection, quality checks, embeddings, matching
├── geofence.py                # Haversine distance, geofence settings, location checks
├── database.py                  # SQLAlchemy models + lightweight self-migration
├── auth.py                        # admin login gate, CSRF tokens, rate limiting
├── config.py                        # every environment-driven setting, in one place
├── requirements.txt
├── Dockerfile                         # Hugging Face Spaces (Docker SDK)
├── README.md                          # This file
├── .dockerignore / .gitignore
├── templates/                           # Jinja2 templates (server-rendered, no JS framework)
│   ├── base.html                          # nav, toast container, shared <head>
│   ├── index.html                          # home — bento-grid overview
│   ├── register.html                        # student registration + webcam capture
│   ├── students.html                          # roster grid, delete, reindex
│   ├── attendance.html                          # check-in page + geolocation capture
│   ├── dashboard.html                             # records, 7-day chart, CSV export
│   ├── settings.html                                # campus map (Leaflet) + geofence config
│   └── login.html                                     # admin password gate
└── static/
    ├── css/style.css                                    # the entire design system
    └── js/
        ├── webcam.js                                        # shared getUserMedia + capture helper
        └── toast.js                                           # renders flashed messages as toasts
```

### `app.py`

The Flask app itself: every route, the security middleware (CSRF check,
security headers, `ProxyFix` for correct client IPs behind Hugging Face's
proxy), and the glue between `face_engine`, `geofence`, `database`, and
`auth`. See [Route reference](#route-reference) for the full route table.

### `face_engine.py`

The recognition pipeline described above. Also owns the on-disk embedding
gallery (`load_gallery` / `save_gallery` / `add_photo_to_gallery` /
`remove_student_from_gallery` / `reindex_gallery`).

### `geofence.py`

`haversine_distance_meters()` (the distance math), `load_settings()` /
`save_settings()` (reads/writes `DATA_DIR/geofence.json`, validating ranges
so an admin can't accidentally set a 0m radius and lock everyone out), and
`check_location()` (the actual per-check-in verdict: allowed / verified /
distance / reason).

### `database.py`

Two tables:

- **`Student`** — `student_id`, `name`, `image_folder`, `created_at`.
- **`AttendanceRecord`** — `student_pk`, `date`, `time`, `confidence`, plus
  `latitude` / `longitude` / `distance_meters` / `location_verified`
  (nullable — only populated when geofencing is on).

Also `run_lightweight_migrations()`, called on every startup: adds any
columns/indexes the current models expect but an older database file
doesn't have yet (plain `ALTER TABLE`/`CREATE INDEX`, no Alembic needed for
a handful of nullable columns). Safe to run repeatedly — it's a no-op once
caught up.

### `auth.py`

Everything access-control related lives here rather than scattered through
`app.py`:

- `login_required` / `is_logged_in` — the admin gate itself. A no-op if
  `ADMIN_PASSWORD` is unset.
- `check_password` — constant-time comparison (`hmac.compare_digest`), so a
  wrong-password response doesn't leak timing information about how many
  characters were correct.
- `RateLimiter` — a small in-memory sliding-window limiter, used for login
  attempts and for `/attendance` (stops both brute-forcing the admin
  password and hammering the CNN with repeated recognition requests).
  **In-memory means per-process** — it matches this app's single gunicorn
  worker (see the Dockerfile) but won't coordinate across multiple
  workers/replicas. A shared store (Redis) would be the real fix at that
  scale; noted here rather than silently assumed away.
- CSRF helpers — a minimal, dependency-free CSRF implementation (a random
  token bound to the session, checked on every POST) instead of pulling in
  Flask-WTF for a handful of forms.

### `config.py`

Every tunable in one place, all environment-variable-driven with sane
defaults. See [Configuration reference](#configuration-reference).

### Templates & static

Server-rendered Jinja2, vanilla CSS and JS — no React/Vue/build step. The
design system (`static/css/style.css`) is a dark, glassmorphic "bento grid"
layout: variable-size translucent tiles establish visual hierarchy (the
biggest tile is always the most important number on that page), an aurora
gradient (violet → fuchsia → amber) marks brand/primary actions, and a
separate teal accent is reserved specifically for location/geofence UI so
it reads as its own semantic category at a glance.

---

## Setup — local development

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:7860**. `tensorflow-cpu`, `keras`, and `mtcnn` are
sizeable downloads (a few hundred MB combined) — first run will take a
minute, and MobileNetV2's ImageNet weights download automatically on first
use of the recognition engine.

Webcam capture and the browser's Geolocation API both need a "secure
context" — most browsers only allow camera/location access on `localhost`
or HTTPS, so `127.0.0.1:7860` works fine for local testing, but you can't
reach it over plain HTTP from another device on your network.

---

## Setup — Hugging Face Spaces

### Quick Start

1. **Create a new Space** with Docker SDK and Python 3.11 runtime
2. **Push this repository** to your Space (the metadata in this README's frontmatter will be automatically detected)
3. **Configure environment variables** in Space Settings → Secrets:
   - `ADMIN_PASSWORD`: (secret) - Strong admin password to protect your data
   - `SECRET_KEY`: (secret) - Random secret key for session security
   - `DATA_DIR`: /data (if using persistent storage) - to persist data across restarts
4. **Wait for build** - First build includes TensorFlow (~5-10 mins), subsequent builds use cache
5. **Configure your campus location** by visiting `/settings` after deployment

### Requirements

- **Runtime**: Python 3.11 (specified in Dockerfile)
- **Hardware**: CPU recommended (GPU not required for MobileNetV2 inference)
- **Storage**: Optional persistent storage for data retention across restarts
- **Visibility**: Private recommended if storing real student data

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ADMIN_PASSWORD` | Yes | Admin password for access control |
| `SECRET_KEY` | Yes | Secret key for session/CSRF protection |
| `DATA_DIR` | No | Set to `/data` if using persistent storage |
| `FLASK_DEBUG` | No | Set to `true` for debugging (not recommended for production) |

### Persistent Storage (Recommended)

For production use, attach persistent storage to preserve:
- Registered students and their photos
- Attendance records
- Geofence settings

To enable:
1. Go to Space Settings → Storage
2. Attach a persistent volume
3. Set `DATA_DIR=/data` in Space Settings → Variables

### Customization

The application supports light/dark theme toggle and customizable geofencing settings accessible via the `/settings` page.

---

## Configuration reference

Every one of these is an environment variable (Hugging Face "Variables and
secrets", or a local `.env`/shell export). Nothing here requires touching
code.

### Core / storage

| Variable | Default | Purpose |
|---|---|---|---|
| `DATA_DIR` | `./data` | Where photos, the DB, and embeddings live |
| `SECRET_KEY` | dev default | Set a real random value in production |
| `PORT` | `7860` | Only used by `python app.py`; gunicorn's `--bind` is separate |
| `FLASK_DEBUG` | `false` | Never enable this on a public deployment |
| `MAX_CONTENT_LENGTH_MB` | `25` | Hard cap on any single request body, in MB |
| `FRAME_ANCESTORS` | `'self' https://huggingface.co https://*.hf.space` | CSP `frame-ancestors` allowlist. Defaults to letting Hugging Face embed the Space in its iframe; set to `'none'` for a standalone deploy that should never be framed |

### Admin access

| Variable | Default | Purpose |
|---|---|---|---|
| `ADMIN_PASSWORD` | *(unset)* | Set to enable the admin login gate |

### Face recognition

| Variable | Default | Purpose |
|---|---|---|---|
| `MATCH_THRESHOLD` | `0.55` | Minimum cosine similarity to accept a match |
| `MATCH_MARGIN` | `0.05` | Best match must beat the runner-up by this much |
| `BLUR_THRESHOLD` | `60` | Laplacian variance floor (lower = more blur allowed) |
| `MIN_FACE_SIZE` | `80` | Minimum detected face size, in pixels |
| `MIN_BRIGHTNESS` / `MAX_BRIGHTNESS` | `40` / `225` | Acceptable mean-brightness range |
| `MAX_PHOTO_UPLOADS` | `10` | Max files per registration upload |
| `MAX_CAPTURED_IMAGES` | `10` | Max webcam captures per registration (server-enforced) |
| `MAX_PHOTO_SIZE` | `5` (MB) | Reserved for future per-file size enforcement |

If real check-ins are being rejected too often, lower `MATCH_THRESHOLD`
slightly; if strangers are being matched, raise it. Re-registering students
with more/varied photos usually helps more than threshold tuning does.

### Geofencing (fallback defaults — see `/settings` for the real admin UI)

| Variable | Default | Purpose |
|---|---|---|---|
| `UNIVERSITY_LAT` / `UNIVERSITY_LNG` | *(unset)* | Initial campus center, before an admin sets one via the map |
| `GEOFENCE_RADIUS_METERS` | `300` | Allowed radius |
| `GEOFENCE_MODE` | `strict` | `strict` (block) or `flag` (allow, mark for review) |
| `GEOFENCE_MAX_ACCURACY_METERS` | `150` | Reject GPS readings less precise than this |

These env vars only matter the very first time the app runs. Once an admin
saves settings via `/settings`, that file (`DATA_DIR/geofence.json`) always
wins over the env vars — so redeploying the container never silently resets
your campus location.

---

## Route reference

| Route | Methods | Auth | Notes |
|---|---|---|---|
| `/` | GET | open | Home / overview |
| `/login` | GET, POST | open | No-op redirect if `ADMIN_PASSWORD` unset |
| `/logout` | GET | open | |
| `/register` | GET, POST | **admin** | |
| `/students` | GET | **admin** | Roster grid |
| `/students/<id>/delete` | POST | **admin** | |
| `/reindex` | POST | **admin** | Rebuilds the embedding gallery from stored photos |
| `/attendance` | GET, POST | open (kiosk) | Rate-limited per IP |
| `/dashboard` | GET | **admin** | |
| `/dashboard/export.csv` | GET | **admin** | |
| `/settings` | GET, POST | **admin** | Campus geofence configuration |
| `/media/<id>/<file>` | GET | conditional | Open only when `ADMIN_PASSWORD` is unset |
| `/healthz` | GET | open | `{"status": "ok"}` for uptime checks |

"open (kiosk)" means `/attendance` is intentionally reachable without
logging in — it's meant to run on a shared device at the entrance, the way
a physical time clock would. Every other route that touches student data or
configuration requires the admin password once one is set.

---

## Feature walkthrough

**Registering a student** (`/register`) — enter an ID and name, then either
capture 3–5 webcam photos from different angles or upload files. Each photo
is quality-checked (face size, blur, brightness) with feedback shown
immediately. The student is recognizable the moment you hit submit — no
training step, no waiting for other students to be added first.

**Taking attendance** (`/attendance`) — capture or upload a photo. If
geofencing is on, the browser's location is requested once (not tracked
continuously) and shown as a status pill before you even submit, so you
know if you're in range. On submit, location is checked first (cheap, fails
fast), then the photo goes through recognition. A student can only be
marked present once per day — enforced both in application logic and by a
database-level unique index, so two near-simultaneous check-ins can't both
slip through as duplicate rows.

**Student roster** (`/students`) — every registered student with their
reference photo, a photo count, and a remove button. "Reindex Photos" walks
every stored photo on disk and rebuilds the embedding gallery from scratch
— useful after manually adding photos to a student's folder, or after
changing detection/quality settings.

**Dashboard** (`/dashboard`) — today's present count, attendance rate, a
count of location-flagged check-ins, a 7-day bar chart, filterable
records, and CSV export (including location columns).

**Campus settings** (`/settings`) — a Leaflet/OpenStreetMap map (no API key
needed). Click anywhere or drag the marker to set the campus center, or hit
"Use My Current Location" if you're standing on campus while configuring
it. Set a radius, choose strict vs. flag mode, and save.

**Theme toggle** — A light/dark theme toggle button is available in the navigation bar for user preference.

---

## Security model

This section exists because "face recognition attendance app" sounds like
it should obviously be secure, and the gap between that assumption and
reality is exactly where real deployments get hurt. Here's what's actually
in place, and why:

**Session & credentials**
-


**Session & credentials**
- Password comparison uses `hmac.compare_digest` (constant-time), not `==`.
- Login attempts are rate-limited (5 per 5 minutes per IP) to blunt
  brute-forcing a single shared password.
- Session cookies are `HttpOnly`, `SameSite=Lax`, and `Secure` outside of
  debug mode.

**CSRF** — every POST route requires a session-bound token, checked in an
`app.before_request` hook. This covers registration, deletion, settings
changes, reindexing, and login itself, closing the "malicious page tricks a
logged-in admin's browser into submitting a form" class of attack.

**Rate limiting** — beyond login, `/attendance` is capped at 20 attempts
per minute per IP, since each attempt runs real CNN inference; without this
an attacker (or a broken client in a retry loop) could turn the check-in
kiosk into a CPU-exhaustion vector.

**Request size limits** — `MAX_CONTENT_LENGTH` caps the whole request body
at the Flask/Werkzeug level (so an oversized request is rejected before
being fully read into memory), and the number of webcam captures accepted
per registration is enforced server-side, not just by the UI's photo-count
limit (which a hand-built request could trivially ignore).

**Photo privacy** — student photos live outside `/static`, served only
through `/media/<id>/<file>`, which is gated behind the admin login once
one is configured, and rejects any path that doesn't survive
`secure_filename`/character-allowlisting unchanged (blocking traversal
attempts like `../../etc/passwd`).

**Security headers** — every response gets `X-Content-Type-Options:
nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy:
strict-origin-when-cross-origin`, and a `Content-Security-Policy` that
explicitly allowlists only the CDNs this app actually loads (fonts, Lucide
icons, Leaflet, Chart.js) rather than allowing arbitrary third-party
scripts. Inline `<script>` blocks are used throughout the templates for
simplicity, which requires `'unsafe-inline'` on `script-src` — a
nonce-based policy would close that specific gap for anyone hardening this
further.

**Database race conditions** — duplicate student IDs and duplicate
same-day attendance records are guarded both at the application level
(check-then-insert, the fast path) and at the database level (unique
constraints, the actual guarantee) — two near-simultaneous requests can't
both slip through the application-level check and create two rows.

**Input validation** — geofence settings reject out-of-range coordinates,
non-finite values (`NaN`/`Infinity` parse successfully in Python's
`float()`, so this needs an explicit check), and radii too small to be
meaningful, so a typo in `/settings` can't silently lock every student out
with no clear error.

**What this deliberately does not include** — full user accounts (it's one
shared admin password by design, matching the original "no login system
needed" brief while still closing the worst gaps for public hosting), and
a nonce-based CSP (the inline-script tradeoff above). Both are reasonable
next steps if this grows into a multi-admin, higher-stakes deployment.

---

## Data & privacy

- Photos, the database, and geofence settings all live under `DATA_DIR`,
  which is both `.gitignore`d and `.dockerignore`d — they will not end up
  in your repo history or Docker image.
- Without `ADMIN_PASSWORD` set, the app runs fully open (registration,
  roster, dashboard, photos) — fine for a local demo, not for a public URL
  holding real student data.
- On Hugging Face Spaces specifically: without a persistent Storage volume,
  the container filesystem resets on restart/rebuild. Attach Storage and
  set `DATA_DIR=/data` if this needs to survive redeploys.
- Consider a **private** Space if this will hold real student data at
  all — Spaces supports private visibility with access control.
- Location data (GPS coordinates, distance from campus) is stored per
  attendance record when geofencing is on. This is meaningfully sensitive
  data about where a real person was standing at a specific time — treat
  the database export with the same care as the photos.

---

## Known limitations

- Browser GPS is spoofable (see [Geofencing](#geofencing) above) — this
  raises the bar, it isn't cryptographic proof of presence.
- The admin gate is one shared password, not per-user accounts with
  individual audit trails.
- Rate limiting is in-memory and per-process — it won't coordinate across
  multiple gunicorn workers or replicas. Fine for the single-worker setup
  in the included Dockerfile; would need a shared store (Redis) to scale
  beyond that.
- Recognition accuracy depends heavily on registration photo quality/
  variety — 3–5 photos from different angles and lighting will noticeably
  outperform a single selfie.
- The Content-Security-Policy allows `'unsafe-inline'` scripts, a
  pragmatic tradeoff for keeping templates simple; a nonce-based policy
  would be stricter.
- Performance may be limited on CPU-only instances when processing multiple
  concurrent face recognition requests.

---

## Troubleshooting

**"No students indexed" / recognition always fails** — check `/students`
to confirm photos were actually saved (a face has to be detected in at
least one photo for a student to appear in the gallery). Try "Reindex
Photos" if you've added images directly to disk.

**Geofencing rejects check-ins that should be valid** — visit `/settings`
and confirm the marker is actually on campus and the radius is generous
enough for GPS drift (150–300m is reasonable to start). Try "flag" mode
temporarily to see actual reported distances on the dashboard without
blocking anyone while you tune it.

**"Too many attempts" on login or attendance** — that's the rate limiter;
wait for the window to pass (5 minutes for login, 1 minute for attendance).

**Students/photos disappeared after a redeploy on Hugging Face** — you
need a persistent Storage volume attached and `DATA_DIR=/data` set; without
both, the container's filesystem resets on every rebuild.

**400 error on a form submission** — almost always an expired/missing CSRF
token, usually because the page was open in a tab for a long time before
submitting. Reload the page and try again.

**Slow performance on Hugging Face Spaces** — CPU instances have limited
resources. Consider optimizing the number of concurrent users or upgrading
the Space hardware if needed.

**Build fails on Hugging Face Spaces** — First builds include TensorFlow
which can take 10-15 minutes. Subsequent builds use cache and are faster.
Check the build logs for specific error messages.