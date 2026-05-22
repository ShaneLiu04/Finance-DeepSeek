#!/usr/bin/env python3
"""
一键启动 API 服务
"""

import sys
import uvicorn
import yaml
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))


def main():
    cfg = yaml.safe_load(open(project_root / "config.yaml", "r", encoding="utf-8"))
    api_cfg = cfg["api"]
    uvicorn.run(
        "finance_deepseek.api.main:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=False,
    )


if __name__ == "__main__":
    main()
