from __future__ import annotations

import hmac

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import get_settings


BAILIAN_TOOL_TOKEN_HEADER_NAME = "X-Zhiwen-Tool-Token"
bailian_tool_token_header = APIKeyHeader(
    name=BAILIAN_TOOL_TOKEN_HEADER_NAME,
    auto_error=False,
    scheme_name="BailianToolToken",
)


async def authorize_bailian_tool_request(
    tool_token: str | None = Security(bailian_tool_token_header),
) -> None:
    configured_token = get_settings().bailian_tool_token.get_secret_value().strip()
    if not configured_token:
        raise HTTPException(status_code=503, detail="bailian_tool_unconfigured")
    if tool_token is None or not hmac.compare_digest(tool_token, configured_token):
        raise HTTPException(status_code=401, detail="bailian_tool_unauthorized")
