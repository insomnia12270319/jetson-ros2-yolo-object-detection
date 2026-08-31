from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLD_DIR = Path(r"C:\Users\Lenovo\Desktop\myphotos")
NEW_DIR = Path(r"C:\Users\Lenovo\Desktop\mynewphotos")

OUTPUT_IMAGE_DIR = PROJECT_ROOT / "dataset" / "custom_finetune" / "images" / "train"
OUTPUT_LABEL_DIR = PROJECT_ROOT / "dataset" / "custom_finetune" / "labels" / "train"

TEST_IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "test"

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# 第二批中，这些是保留的 test，绝不能进入专项训练
test_stems = {p.stem for p in TEST_IMAGE_DIR.glob("*.jpg")}


def copy_dataset(source_dir, exclude_stems=None):
    label_dir = source_dir / "yolo_labels"
    count = 0

    for label_path in label_dir.glob("*.txt"):
        stem = label_path.stem

        if exclude_stems and stem in exclude_stems:
            continue

        image_path = source_dir / f"{stem}.jpg"

        if not image_path.exists():
            print("Missing image:", image_path)
            continue

        shutil.copy2(image_path, OUTPUT_IMAGE_DIR / image_path.name)
        shutil.copy2(label_path, OUTPUT_LABEL_DIR / label_path.name)
        count += 1

    return count


old_count = copy_dataset(OLD_DIR)
new_count = copy_dataset(NEW_DIR, exclude_stems=test_stems)

print("Old custom images:", old_count)
print("New hardcase train images:", new_count)
print("Total custom finetune images:", old_count + new_count)