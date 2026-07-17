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