from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据集目录
DATASET_DIR = PROJECT_ROOT / "dataset"

# 训练两个类别
CLASS_MAP = {
    "mouse": 0,
    "cell phone": 1,
}

print("Project root:", PROJECT_ROOT)
print("Dataset directory:", DATASET_DIR)
print("Classes:", CLASS_MAP)