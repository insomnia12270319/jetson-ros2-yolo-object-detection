from pathlib import Path
import shutil
#捕捉困难图片，放大权重
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# v8 作为基础
BASE_ROOT = PROJECT_ROOT / "dataset" / "mixed_domain_v8"

SOURCE_DIR = Path(r"C:\Users\Lenovo\Desktop\mynewphotos")
SOURCE_LABEL_DIR = SOURCE_DIR / "yolo_labels"

TEST_IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "test"

OUT_ROOT = PROJECT_ROOT / "dataset" / "mixed_hardcase_v9"
OUT_IMAGE_DIR = OUT_ROOT / "images" / "train"
OUT_LABEL_DIR = OUT_ROOT / "labels" / "train"

if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

shutil.copytree(BASE_ROOT, OUT_ROOT)

test_stems = {
    p.stem for p in TEST_IMAGE_DIR.glob("*.jpg")
}

added = 0

#  只加入 hard_mouse_001 ~ hard_mouse_014
# 每张出现 4 次

for i in range(1, 15):

    stem = f"hard_mouse_{i:03d}"

    image_path = SOURCE_DIR / f"{stem}.jpg"
    label_path = SOURCE_LABEL_DIR / f"{stem}.txt"

    if stem in test_stems:
        print("SKIP TEST:", stem)
        continue

    if not image_path.exists():
        print("Missing image:", image_path)
        continue

    if not label_path.exists():
        print("Missing label:", label_path)
        continue

    # 每张 hardcase 复制 4 份
    for repeat in range(1, 5):

        image_name = f"hardv9_{repeat}_{image_path.name}"
        label_name = f"hardv9_{repeat}_{label_path.name}"

        shutil.copy2(
            image_path,
            OUT_IMAGE_DIR / image_name
        )

        shutil.copy2(
            label_path,
            OUT_LABEL_DIR / label_name
        )

        added += 1


print()
print("mixed_hardcase_v9")
print("============================")
print("Hardcase source images: 14")
print("Hardcase copies added:", added)
print(
    "Final images:",
    len(list(OUT_IMAGE_DIR.glob("*.jpg")))
)
print(
    "Final labels:",
    len(list(OUT_LABEL_DIR.glob("*.txt")))
)