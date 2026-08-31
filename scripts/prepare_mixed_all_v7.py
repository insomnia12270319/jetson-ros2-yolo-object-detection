from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V5_ROOT = PROJECT_ROOT / "dataset" / "mixed_all_v5"
OUT_ROOT = PROJECT_ROOT / "dataset" / "mixed_all_v7"

TEST_IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "test"

SOURCES = [
    ("old", Path(r"C:\Users\Lenovo\Desktop\myphotos")),
    ("new", Path(r"C:\Users\Lenovo\Desktop\mynewphotos")),
]

# 先完整复制 v5
if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

shutil.copytree(V5_ROOT, OUT_ROOT)

OUT_IMAGE_DIR = OUT_ROOT / "images" / "train"
OUT_LABEL_DIR = OUT_ROOT / "labels" / "train"

test_stems = {p.stem for p in TEST_IMAGE_DIR.glob("*.jpg")}

mouse_only_items = []

# 找出所有自采 mouse-only 图片
for prefix, source_dir in SOURCES:

    label_dir = source_dir / "yolo_labels"

    for label_path in sorted(label_dir.glob("*.txt")):

        stem = label_path.stem

        if stem in test_stems:
            continue

        image_path = source_dir / f"{stem}.jpg"

        if not image_path.exists():
            continue

        classes = {
            line.split()[0]
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }

        if classes == {"0"}:
            mouse_only_items.append(
                (prefix, image_path, label_path)
            )

# 固定随机种子，每次结果一致
rng = random.Random(42)
rng.shuffle(mouse_only_items)

# 只选40张，再额外复制一次
selected = mouse_only_items[:40]

for index, (prefix, image_path, label_path) in enumerate(
    selected, start=1
):

    image_name = (
        f"mousemid_{index}_{prefix}_{image_path.name}"
    )

    label_name = (
        f"mousemid_{index}_{prefix}_{label_path.name}"
    )

    shutil.copy2(
        image_path,
        OUT_IMAGE_DIR / image_name
    )

    shutil.copy2(
        label_path,
        OUT_LABEL_DIR / label_name
    )

# 统计
mouse_instances = 0
phone_instances = 0

labels = list(OUT_LABEL_DIR.glob("*.txt"))

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

print("Extra mouse copies:", len(selected))
print("Final images:", len(list(OUT_IMAGE_DIR.glob("*.jpg"))))
print("Mouse instances:", mouse_instances)
print("Phone instances:", phone_instances)