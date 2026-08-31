from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "train"
BASE_LABEL_DIR = PROJECT_ROOT / "dataset" / "labels" / "train"

TEST_IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "test"

SOURCES = [
    ("old", Path(r"C:\Users\Lenovo\Desktop\myphotos")),
    ("new", Path(r"C:\Users\Lenovo\Desktop\mynewphotos")),
]

# v6 输出目录
OUT_ROOT = PROJECT_ROOT / "dataset" / "mixed_all_v6"
OUT_IMAGE_DIR = OUT_ROOT / "images" / "train"
OUT_LABEL_DIR = OUT_ROOT / "labels" / "train"

if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

OUT_IMAGE_DIR.mkdir(parents=True)
OUT_LABEL_DIR.mkdir(parents=True)

# --------------------------------------------------
# 1. 复制当前完整主训练集
# --------------------------------------------------

for image in BASE_IMAGE_DIR.glob("*.jpg"):
    shutil.copy2(image, OUT_IMAGE_DIR / image.name)

for label in BASE_LABEL_DIR.glob("*.txt"):
    shutil.copy2(label, OUT_LABEL_DIR / label.name)

base_stems = {p.stem for p in BASE_IMAGE_DIR.glob("*.jpg")}
test_stems = {p.stem for p in TEST_IMAGE_DIR.glob("*.jpg")}

# 安全检查：test 绝不能已经存在 train
leaked = base_stems & test_stems

if leaked:
    print("WARNING: test leakage found:", leaked)
else:
    print("Test leakage check: OK")

added_unique = 0
mouse_boost_copies = 0

# --------------------------------------------------
# 2. 加入全部自采图片
# --------------------------------------------------

for prefix, source_dir in SOURCES:

    label_dir = source_dir / "yolo_labels"

    for label_path in sorted(label_dir.glob("*.txt")):

        stem = label_path.stem

        # 8 张最终 test 绝对排除
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

        # 如果这张自采图还没包含在主训练集，加入一次
        if stem not in base_stems:

            image_name = f"{prefix}_{image_path.name}"
            label_name = f"{prefix}_{label_path.name}"

            shutil.copy2(
                image_path,
                OUT_IMAGE_DIR / image_name
            )

            shutil.copy2(
                label_path,
                OUT_LABEL_DIR / label_name
            )

            added_unique += 1

        # --------------------------------------------------
        # mouse-only 图片额外复制 3 次
        # 提高困难 mouse 场景的训练权重
        # --------------------------------------------------

        if classes == {"0"}:

            for repeat in range(1, 4):

                image_name = (
                    f"mouseboost{repeat}_{prefix}_{image_path.name}"
                )

                label_name = (
                    f"mouseboost{repeat}_{prefix}_{label_path.name}"
                )

                shutil.copy2(
                    image_path,
                    OUT_IMAGE_DIR / image_name
                )

                shutil.copy2(
                    label_path,
                    OUT_LABEL_DIR / label_name
                )

                mouse_boost_copies += 1


# --------------------------------------------------
# 3. 最终统计
# --------------------------------------------------

images = list(OUT_IMAGE_DIR.glob("*.jpg"))
labels = list(OUT_LABEL_DIR.glob("*.txt"))

mouse_instances = 0
phone_instances = 0

for label in labels:

    for line in label.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        class_id = line.split()[0]

        if class_id == "0":
            mouse_instances += 1

        elif class_id == "1":
            phone_instances += 1


print()
print("============================")
print("mixed_all_v6")
print("============================")

print("Base train:", len(base_stems))
print("New unique custom:", added_unique)
print("Mouse boost copies:", mouse_boost_copies)

print()
print("Final images:", len(images))
print("Final labels:", len(labels))

print()
print("Mouse instances:", mouse_instances)
print("Phone instances:", phone_instances)