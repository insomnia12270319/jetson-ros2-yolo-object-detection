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
TRAIN_PER_CLASS = 1000
VAL_PER_CLASS = 80
RANDOM_SEED = 42
print("Project root:", PROJECT_ROOT)
print("Dataset directory:", DATASET_DIR)
print("Classes:", CLASS_MAP)
import zipfile
import random
import requests
ZIP_PATH = DATASET_DIR / "coco_raw" / "coco2017labels.zip"

TARGET_CLASS_IDS = {64, 67}  # COCO: 64=mouse, 67=cell phone
COCO_TO_CUSTOM = {
    64: 0,  # mouse
    67: 1,  # cell phone -> phone
}
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    train_mouse = []
    train_phone = []
    val_mouse = []
    val_phone = []

    for name in zf.namelist():
        if not name.endswith(".txt"):
            continue

        if "/train2017/" not in name and "/val2017/" not in name:
            continue

        text = zf.read(name).decode("utf-8")
        class_ids = {
            int(line.split()[0])
            for line in text.splitlines()
            if line.strip()
        }

        if "/train2017/" in name:
            if 64 in class_ids:
                train_mouse.append(name)
            if 67 in class_ids:
                train_phone.append(name)

        elif "/val2017/" in name:
            if 64 in class_ids:
                val_mouse.append(name)
            if 67 in class_ids:
                val_phone.append(name)

print("Train mouse:", len(train_mouse))
print("Train phone:", len(train_phone))
print("Val mouse:", len(val_mouse))
print("Val phone:", len(val_phone))
rng = random.Random(RANDOM_SEED)

# 从每个类别中固定随机抽取指定数量
selected_train_mouse = rng.sample(
    train_mouse,
    min(TRAIN_PER_CLASS, len(train_mouse))
)

selected_train_phone = rng.sample(
    train_phone,
    min(TRAIN_PER_CLASS, len(train_phone))
)

selected_val_mouse = rng.sample(
    val_mouse,
    min(VAL_PER_CLASS, len(val_mouse))
)

selected_val_phone = rng.sample(
    val_phone,
    min(VAL_PER_CLASS, len(val_phone))
)

# 同一张图可能同时包含 mouse 和 phone，所以需要去重
selected_train = sorted(
    set(selected_train_mouse) | set(selected_train_phone)
)

selected_val = sorted(
    set(selected_val_mouse) | set(selected_val_phone)
)

print("Selected train mouse:", len(selected_train_mouse))
print("Selected train phone:", len(selected_train_phone))
print("Unique train images:", len(selected_train))

print("Selected val mouse:", len(selected_val_mouse))
print("Selected val phone:", len(selected_val_phone))
print("Unique val images:", len(selected_val))
# 输出目录
TRAIN_LABEL_DIR = DATASET_DIR / "labels" / "train"
VAL_LABEL_DIR = DATASET_DIR / "labels" / "val"

TRAIN_LABEL_DIR.mkdir(parents=True, exist_ok=True)
VAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)


def convert_label(text):
    converted_lines = []

    for line in text.splitlines():
        if not line.strip():
            continue

        parts = line.split()
        coco_class_id = int(parts[0])

        # 只保留 mouse 和 cell phone
        if coco_class_id not in COCO_TO_CUSTOM:
            continue

        # 替换为我们自己的类别编号
        parts[0] = str(COCO_TO_CUSTOM[coco_class_id])

        converted_lines.append(" ".join(parts))

    return "\n".join(converted_lines) + "\n"


with zipfile.ZipFile(ZIP_PATH, "r") as zf:

    for label_name in selected_train:
        text = zf.read(label_name).decode("utf-8")
        converted = convert_label(text)

        output_path = TRAIN_LABEL_DIR / Path(label_name).name
        output_path.write_text(converted, encoding="utf-8")

    for label_name in selected_val:
        text = zf.read(label_name).decode("utf-8")
        converted = convert_label(text)

        output_path = VAL_LABEL_DIR / Path(label_name).name
        output_path.write_text(converted, encoding="utf-8")


print("Generated train labels:", len(selected_train))
print("Generated val labels:", len(selected_val))
# 图片输出目录
TRAIN_IMAGE_DIR = DATASET_DIR / "images" / "train"
VAL_IMAGE_DIR = DATASET_DIR / "images" / "val"

TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
VAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def download_images(label_names, split, output_dir):
    total = len(label_names)

    for index, label_name in enumerate(label_names, start=1):
        image_name = Path(label_name).stem + ".jpg"
        save_path = output_dir / image_name

        # 已经下载过的图片直接跳过
        if save_path.exists():
            continue

        url = f"http://images.cocodataset.org/{split}2017/{image_name}"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            save_path.write_bytes(response.content)

        except requests.RequestException as e:
            print(f"Failed: {image_name}")
            print(e)
            continue

        if index % 50 == 0 or index == total:
            print(f"{split}: {index}/{total}")


print("\nDownloading training images...")
download_images(
    selected_train,
    "train",
    TRAIN_IMAGE_DIR
)

print("\nDownloading validation images...")
download_images(
    selected_val,
    "val",
    VAL_IMAGE_DIR
)

print("\nImage download finished.")