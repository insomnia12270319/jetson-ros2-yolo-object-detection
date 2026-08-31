from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCES = [
    ("old", Path(r"C:\Users\Lenovo\Desktop\myphotos")),
    ("new", Path(r"C:\Users\Lenovo\Desktop\mynewphotos")),
]

# 原来保留的 8 张最终测试图
TEST_IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "test"

# 新建 v3，不覆盖旧数据
OUT_ROOT = PROJECT_ROOT / "dataset" / "custom_balanced_v3"

TRAIN_IMG = OUT_ROOT / "images" / "train"
TRAIN_LBL = OUT_ROOT / "labels" / "train"

VAL_IMG = OUT_ROOT / "images" / "val"
VAL_LBL = OUT_ROOT / "labels" / "val"

# 如果以前运行过 v3，先清空，防止旧文件残留
if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

for folder in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    folder.mkdir(parents=True, exist_ok=True)

# 固定随机种子，保证每次划分一致
rng = random.Random(42)

# 8 张 test 永远排除
test_stems = {p.stem for p in TEST_IMAGE_DIR.glob("*.jpg")}

groups = {
    "mouse_only": [],
    "phone_only": [],
    "both": [],
}

# --------------------------------------------------
# 读取两批自采数据
# --------------------------------------------------
for prefix, source_dir in SOURCES:

    label_dir = source_dir / "yolo_labels"

    for label_path in sorted(label_dir.glob("*.txt")):

        stem = label_path.stem

        # test 禁止进入 train / val
        if stem in test_stems:
            continue

        image_path = source_dir / f"{stem}.jpg"

        if not image_path.exists():
            print("Missing image:", image_path)
            continue

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        classes = {
            line.split()[0]
            for line in lines
        }

        item = {
            "prefix": prefix,
            "image": image_path,
            "label": label_path,
        }

        if "0" in classes and "1" in classes:
            groups["both"].append(item)

        elif "0" in classes:
            groups["mouse_only"].append(item)

        elif "1" in classes:
            groups["phone_only"].append(item)


print("Available after excluding test:")
for name, items in groups.items():
    print(f"{name}: {len(items)}")


# --------------------------------------------------
# 真正做平衡采样
# --------------------------------------------------

# mouse-only 太多，只随机取 30 张
rng.shuffle(groups["mouse_only"])
selected_mouse = groups["mouse_only"][:30]

# phone-only 全部保留
selected_phone = list(groups["phone_only"])

# mouse + phone 同框全部保留
selected_both = list(groups["both"])

selected_groups = {
    "mouse_only": selected_mouse,
    "phone_only": selected_phone,
    "both": selected_both,
}

print("\nSelected:")
for name, items in selected_groups.items():
    print(f"{name}: {len(items)}")


# --------------------------------------------------
# 每组约 20% 做 validation
# --------------------------------------------------

train_items = []
val_items = []

for group_name, items in selected_groups.items():

    rng.shuffle(items)

    val_count = max(
        1,
        round(len(items) * 0.20)
    )

    val_part = items[:val_count]
    train_part = items[val_count:]

    val_items.extend(val_part)
    train_items.extend(train_part)

    print(
        f"{group_name}: "
        f"train={len(train_part)}, "
        f"val={len(val_part)}"
    )


# --------------------------------------------------
# 复制文件
# --------------------------------------------------

def copy_items(items, image_out, label_out):

    for item in items:

        prefix = item["prefix"]
        image_path = item["image"]
        label_path = item["label"]

        # 前缀避免 old/new 出现同名文件
        image_name = f"{prefix}_{image_path.name}"
        label_name = f"{prefix}_{label_path.name}"

        shutil.copy2(
            image_path,
            image_out / image_name
        )

        shutil.copy2(
            label_path,
            label_out / label_name
        )


copy_items(train_items, TRAIN_IMG, TRAIN_LBL)
copy_items(val_items, VAL_IMG, VAL_LBL)


# --------------------------------------------------
# 统计最终实例数量
# --------------------------------------------------

def count_instances(items):

    mouse_count = 0
    phone_count = 0

    for item in items:

        for line in item["label"].read_text(
            encoding="utf-8"
        ).splitlines():

            if not line.strip():
                continue

            class_id = line.split()[0]

            if class_id == "0":
                mouse_count += 1

            elif class_id == "1":
                phone_count += 1

    return mouse_count, phone_count


train_mouse, train_phone = count_instances(train_items)
val_mouse, val_phone = count_instances(val_items)


print("\n==============================")
print("Final dataset")
print("==============================")

print("Train images:", len(train_items))
print("Val images:", len(val_items))
print("Excluded test:", len(test_stems))

print()
print("Train mouse instances:", train_mouse)
print("Train phone instances:", train_phone)

print()
print("Val mouse instances:", val_mouse)
print("Val phone instances:", val_phone)

print()
print("Saved to:", OUT_ROOT)