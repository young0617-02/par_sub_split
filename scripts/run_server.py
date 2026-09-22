"""启动 HTTP 服务（等价于 python -m uvicorn app.api.main:app --reload）。

用法：
    python scripts/run_server.py                # 默认 0.0.0.0:8000
    python scripts/run_server.py --port 8001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 FastAPI 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    print(f"服务启动中： http://{args.host}:{args.port}/  （可视化调试页）")
    print(f"接口文档：   http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
