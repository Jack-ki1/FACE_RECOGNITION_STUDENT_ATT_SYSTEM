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
"""

from functools import wraps
from flask import session, redirect, url_for, request

import config


def is_logged_in():
    return (not config.AUTH_ENABLED) or session.get("logged_in", False)


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not is_logged_in():
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapped
