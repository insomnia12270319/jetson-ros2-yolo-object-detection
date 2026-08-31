from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCES = [
    ("old", Path(r"C:\Users\Lenovo\Desktop\myphotos")),
    ("new", Path(r"C:\Users\Lenovo\Desktop\mynewphotos")),
]

TEST_DIR = PROJECT_ROOT / "dataset" / "images" / "test"

OUT_ROOT = PROJECT_ROOT / "dataset" / "custom_balanced"
TRAIN_IMG = OUT_ROOT / "images" / "train"
TRAIN_LBL = OUT_ROOT / "labels" / "train"
VAL_IMG = OUT_ROOT / "images" / "val"
VAL_LBL = OUT_ROOT / "labels" / "val"

for p in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    p.mkdir(parents=True, exist_ok=True)

test_stems = {p.stem for p in TEST_DIR.glob("*.jpg")}

groups = {
    "mouse_only": [],
    "phone_only": [],
    "both": [],
}

for prefix, source in SOURCES:
    label_dir = source / "yolo_labels"

    for label in label_dir.glob("*.txt"):
        stem = label.stem

        if stem in test_stems:
            continue

        image = source / f"{stem}.jpg"
        if not image.exists():
            continue

        classes = {
            line.split()[0]
            for line in label.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        item = (prefix, image, label)

        if "0" in classes and "1" in classes:
            groups["both"].append(item)
        elif "0" in classes:
            groups["mouse_only"].append(item)
        elif "1" in classes:
            groups["phone_only"].append(item)

rng = random.Random(42)

train_items = []
val_items = []

# 每种场景约20%留作自采验证集
for group_name, items in groups.items():
    rng.shuffle(items)

    val_count = max(1, round(len(items) * 0.20))
    val_items.extend(items[:val_count])
    train_items.extend(items[val_count:])

    print(
        group_name,
        "total:", len(items),
        "train:", len(items) - val_count,
        "val:", val_count
    )


def copy_items(items, image_out, label_out):
    for prefix, image, label in items:
        # 加前缀，避免两个文件夹万一有同名图片
        new_name = f"{prefix}_{image.name}"
        new_label = f"{prefix}_{label.name}"

        shutil.copy2(image, image_out / new_name)
        shutil.copy2(label, label_out / new_label)


copy_items(train_items, TRAIN_IMG, TRAIN_LBL)
copy_items(val_items, VAL_IMG, VAL_LBL)

print()
print("Custom train:", len(train_items))
print("Custom val:", len(val_items))
print("Excluded test:", len(test_stems))