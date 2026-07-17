"""
auth.py
-------
A deliberately minimal auth layer: a single shared admin password, checked
via a Flask session flag. No user accounts, no database table -- this is not
meant to replace real auth for a multi-admin institution, just to stop a
public Hugging Face Space URL from being a wide-open door to student data.

If config.ADMIN_PASSWORD is empty (the default), `login_required` is a no-op
and every route behaves exactly as if this file didn't exist -- matching the
original "no login system needed" brief for local/demo use. Set ADMIN_PASSWORD
as an environment variable (an HF Space "secret") to turn the gate on.

This file also carries three small pieces of security hardening added during
a later audit pass, kept here rather than scattered across app.py:

  1. `check_password` -- constant-time password comparison. A plain `==`
     comparison on strings returns as soon as it finds a mismatched
     character, which leaks (via response timing) how many leading
     characters of a guess were correct. Not a huge risk for a low-traffic
     app, but it costs nothing to close.

  2. `RateLimiter` -- a small in-memory sliding-window limiter. Used to
     slow down login brute-forcing and to stop `/attendance` from being
     hammered into running expensive CNN inference in a loop. It's
     per-process/in-memory by design, which matches this app's single
     gunicorn worker (see Dockerfile) -- it will NOT coordinate across
     multiple workers/replicas. Documented as a known limitation in the
     README; a shared store (Redis) would be the real fix at that scale.

  3. CSRF token helpers -- a minimal, dependency-free CSRF protection
     (session-bound random token, checked on every POST) rather than
     pulling in Flask-WTF for a handful of forms.
"""

import hmac
import secrets
import time
from collections import defaultdict
from functools import wraps
from flask import session, redirect, url_for, request, abort

import config


# ---------------------------------------------------------------------------
# Login gate
# ---------------------------------------------------------------------------
def is_logged_in():
    return (not config.AUTH_ENABLED) or session.get("logged_in", False)


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not is_logged_in():
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapped


def check_password(submitted):
    """Constant-time comparison against the configured admin password."""
    if not config.ADMIN_PASSWORD:
        return False
    return hmac.compare_digest(submitted, config.ADMIN_PASSWORD)


# ---------------------------------------------------------------------------
# Rate limiting (in-memory, per-process -- see module docstring)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, max_attempts, window_seconds):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts = defaultdict(list)

    def _prune(self, key, now):
        cutoff = now - self.window_seconds
        self._attempts[key] = [t for t in self._attempts[key] if t > cutoff]

    def is_limited(self, key):
        now = time.time()
        self._prune(key, now)
        return len(self._attempts[key]) >= self.max_attempts

    def record(self, key):
        self._attempts[key].append(time.time())

    def reset(self, key):
        self._attempts.pop(key, None)


login_limiter = RateLimiter(max_attempts=5, window_seconds=300)      # 5 tries / 5 min
attendance_limiter = RateLimiter(max_attempts=20, window_seconds=60)  # 20 tries / min


def client_ip():
    """
    request.remote_addr, trusting X-Forwarded-For as set up by ProxyFix in
    app.py. Falls back to 'unknown' rather than crashing if called outside
    a request context.
    """
    try:
        return request.remote_addr or "unknown"
    except RuntimeError:
        return "unknown"


# ---------------------------------------------------------------------------
# CSRF protection (session-bound token, no external dependency)
# ---------------------------------------------------------------------------
def get_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]


def csrf_valid(submitted_token):
    session_token = session.get("csrf_token")
    if not session_token or not submitted_token:
        return False
    return hmac.compare_digest(session_token, submitted_token)


def csrf_protect():
    """Call from an app.before_request hook. Aborts 400 on a bad/missing token."""
    if request.method == "POST":
        if not csrf_valid(request.form.get("csrf_token", "")):
            abort(400, description="Your session expired or the form was submitted from an untrusted source — please reload the page and try again.")
