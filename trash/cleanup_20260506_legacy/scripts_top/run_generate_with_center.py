"""
入口脚本: 使用训练好的 center token 生成 PTBXL 风格 ECG

Usage:
    python scripts/run_generate_with_center.py --center_token_path checkpoints/center_token/center_token_best.pth
    python scripts/run_generate_with_center.py --center_token_path checkpoints/center_token/center_token_best.pth --generate_baseline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from center_token.generate import main

if __name__ == "__main__":
    main()
