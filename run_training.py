#!/usr/bin/env python3
"""
一键启动 QLoRA 训练
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from finance_deepseek.training.qlora_trainer import main

if __name__ == "__main__":
    main()
