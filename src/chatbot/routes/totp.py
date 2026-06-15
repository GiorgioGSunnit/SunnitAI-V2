"""2FA TOTP endpoints — enroll, enable, disable."""

import pyotp
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...db.base import get_db
from .auth import get_current_user
from ...db.models import User

router = APIRouter(prefix="/auth/me/2fa", tags=["2fa"])

APP_NAME = "Astrea"


class CodeRequest(BaseModel):
    code: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/enroll")
def enroll(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Step 1 — generate a TOTP secret and return the provisioning URI.

    The user scans the URI with their authenticator app, then calls /enable
    with a valid code to activate 2FA. The secret is stored but 2FA is not
    active until /enable succeeds.
    """
    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(name=current_user.email, issuer_name=APP_NAME)

    current_user.totp_secret = secret
    db.commit()

    return {
        "secret": secret,
        "uri": uri,
        "instructions": (
            "Open Google Authenticator or Microsoft Authenticator, "
            "tap '+' → 'Enter setup key', and paste the secret. "
            "Then call POST /auth/me/2fa/enable with the 6-digit code to activate."
        ),
    }


@router.post("/enable")
def enable_2fa(
    request: CodeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Step 2 — verify the first code from the app and activate 2FA."""
    if not current_user.totp_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No enrollment in progress. Call /auth/me/2fa/enroll first.",
        )

    totp = pyotp.TOTP(current_user.totp_secret)
    if not totp.verify(request.code, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid code. Make sure your device clock is correct and try again.",
        )

    current_user.totp_enabled = True
    db.commit()

    return {"message": "2FA enabled successfully. Future logins will require a code."}


@router.post("/disable")
def disable_2fa(
    request: CodeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Disable 2FA — requires a valid code as confirmation."""
    if not current_user.totp_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is not enabled on this account.",
        )

    totp = pyotp.TOTP(current_user.totp_secret)
    if not totp.verify(request.code, valid_window=1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid code.",
        )

    current_user.totp_enabled = False
    current_user.totp_secret = None
    db.commit()

    return {"message": "2FA disabled successfully."}
