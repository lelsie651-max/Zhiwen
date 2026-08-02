from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, is_dataclass
import json
import uuid

from app.core.database import get_async_session_factory
from app.services.bailian_demo_seed import seed_bailian_demo


def _to_jsonable(value):
    if isinstance(value, uuid.UUID):
        return str(value)
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


async def _main_async(confirm_local_demo: bool) -> int:
    if not confirm_local_demo:
        raise SystemExit("bailian_demo_seed_confirmation_required")
    result = await seed_bailian_demo(get_async_session_factory())
    print(json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed local Bailian demo data.")
    parser.add_argument(
        "--confirm-local-demo",
        action="store_true",
        help="Allow writing local demo data.",
    )
    args = parser.parse_args()
    return asyncio.run(_main_async(confirm_local_demo=args.confirm_local_demo))


if __name__ == "__main__":
    raise SystemExit(main())
