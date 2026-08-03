from __future__ import annotations

import hmac

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings


bailian_tool_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="BailianToolBearer",
    bearerFormat="opaque shared token",
)


async def authorize_bailian_tool_request(
    authorization: HTTPAuthorizationCredentials | None = Security(bailian_tool_bearer),
) -> None:
    configured_token = get_settings().bailian_tool_token.get_secret_value().strip()
    if not configured_token:
        raise HTTPException(status_code=503, detail="bailian_tool_unconfigured")
    if (
        authorization is None
        or authorization.scheme.lower() != "bearer"
        or not authorization.credentials
        or not hmac.compare_digest(authorization.credentials, configured_token)
    ):
        raise HTTPException(status_code=401, detail="bailian_tool_unauthorized")
