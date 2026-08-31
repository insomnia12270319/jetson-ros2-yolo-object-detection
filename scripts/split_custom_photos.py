from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path(r"C:\Users\Lenovo\Desktop\mynewphotos")
LABEL_DIR = SOURCE_DIR / "yolo_labels"

TRAIN_IMAGE_DIR = PROJECT_ROOT / "dataset/images/train"
TRAIN_LABEL_DIR = PROJECT_ROOT / "dataset/labels/train"
TEST_IMAGE_DIR = PROJECT_ROOT / "dataset/images/test"
TEST_LABEL_DIR = PROJECT_ROOT / "dataset/labels/test"

labels = sorted(LABEL_DIR.glob("*.txt"))

random.seed(42)
random.shuffle(labels)

train_labels = labels[:20]
test_labels = labels[20:28]

def copy_pair(label, image_dir, label_dir):
    stem = label.stem
    image = SOURCE_DIR / f"{stem}.jpg"

    shutil.copy2(image, image_dir / image.name)
    shutil.copy2(label, label_dir / label.name)

for label in train_labels:
    copy_pair(label, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)

for label in test_labels:
    copy_pair(label, TEST_IMAGE_DIR, TEST_LABEL_DIR)

print("Train added:", len(train_labels))
print("Test added:", len(test_labels))