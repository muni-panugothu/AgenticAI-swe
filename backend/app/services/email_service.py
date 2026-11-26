"""Transactional email delivery (SMTP). Never logs secrets or reset tokens in full."""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

from app.core.config import get_settings

logger = logging.getLogger("agentic.email")


def send_password_reset_email(*, to_email: str, reset_url: str) -> bool:
    """
    Send a password-reset email. Returns True when the message was accepted
    by the transport. Failures are logged without the token.
    """
    settings = get_settings()
    subject = "Reset your Soft Life password"
    body = (
        "We received a request to reset your password.\n\n"
        f"Open this link to choose a new password (expires in "
        f"{settings.password_reset_ttl_minutes} minutes):\n\n"
        f"{reset_url}\n\n"
        "If you did not request this, you can ignore this email.\n"
        "This link can be used only once.\n"
    )

    if not settings.smtp_host or not settings.smtp_from:
        # Dev / unconfigured: do not fail the forgot-password API, but never
        # return the URL to the client (avoids account enumeration leaks).
        logger.warning(
            "SMTP not configured — password reset email skipped for %s",
            _redact_email(to_email),
        )
        if settings.password_reset_log_url_in_dev:
            logger.info("DEV password reset URL for %s: %s", _redact_email(to_email), reset_url)
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from
    msg["To"] = to_email
    msg.set_content(body)

    try:
        if settings.smtp_use_tls:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                if settings.smtp_username:
                    smtp.login(settings.smtp_username, settings.smtp_password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=20) as smtp:
                if settings.smtp_username:
                    smtp.login(settings.smtp_username, settings.smtp_password)
                smtp.send_message(msg)
        logger.info("Password reset email queued for %s", _redact_email(to_email))
        return True
    except Exception as exc:
        logger.error(
            "Failed to send password reset email to %s: %s",
            _redact_email(to_email),
            type(exc).__name__,
        )
        return False


def _redact_email(email: str) -> str:
    parts = email.split("@")
    if len(parts) != 2:
        return "***"
    local, domain = parts
    if len(local) <= 2:
        return f"*@{domain}"
    return f"{local[0]}***{local[-1]}@{domain}"
