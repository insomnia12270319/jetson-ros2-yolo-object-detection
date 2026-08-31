from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# v5 数据集作为基础
V5_ROOT = PROJECT_ROOT / "dataset" / "mixed_all_v5"

SOURCE_DIR = Path(r"C:\Users\Lenovo\Desktop\mynewphotos")
SOURCE_LABEL_DIR = SOURCE_DIR / "yolo_labels"

TEST_IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "test"

OUT_ROOT = PROJECT_ROOT / "dataset" / "mixed_domain_v8"
OUT_IMAGE_DIR = OUT_ROOT / "images" / "train"
OUT_LABEL_DIR = OUT_ROOT / "labels" / "train"

#  完整复制 v5

if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

shutil.copytree(V5_ROOT, OUT_ROOT)

test_stems = {
    p.stem for p in TEST_IMAGE_DIR.glob("*.jpg")
}

v5_image_dir = V5_ROOT / "images" / "train"

v5_stems = {
    p.stem for p in v5_image_dir.glob("*.jpg")
}

added = {
    "mouse_only": 0,
    "phone_only": 0,
    "both": 0,
    "negative": 0,
}

#  加入 v5 之后真正新增的数据

for image_path in sorted(SOURCE_DIR.glob("*.jpg")):

    stem = image_path.stem

    # 永久 test 不允许进入训练
    if stem in test_stems:
        continue

    label_path = SOURCE_LABEL_DIR / f"{stem}.txt"

    if not label_path.exists():
        continue

    # 判断该图片是否已经在 v5 里
    already_in_v5 = any(
        old == stem or old.endswith("_" + stem)
        for old in v5_stems
    )

    if already_in_v5:
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

    if not classes:
        kind = "negative"
    elif classes == {"0"}:
        kind = "mouse_only"
    elif classes == {"1"}:
        kind = "phone_only"
    elif classes == {"0", "1"}:
        kind = "both"
    else:
        continue

    image_name = f"domainv8_{image_path.name}"
    label_name = f"domainv8_{label_path.name}"

    shutil.copy2(
        image_path,
        OUT_IMAGE_DIR / image_name
    )

    shutil.copy2(
        label_path,
        OUT_LABEL_DIR / label_name
    )

    added[kind] += 1



#  最终 test 泄漏检查


train_stems = {
    p.stem for p in OUT_IMAGE_DIR.glob("*.jpg")
}

leaks = []

for test_stem in test_stems:
    for train_stem in train_stems:
        if train_stem == test_stem or train_stem.endswith(
            "_" + test_stem
        ):
            leaks.append(test_stem)

print()
print("============================")
print("mixed_domain_v8")

print("Added:", added)
print(
    "Final images:",
    len(list(OUT_IMAGE_DIR.glob("*.jpg")))
)
print(
    "Final labels:",
    len(list(OUT_LABEL_DIR.glob("*.txt")))
)

if leaks:
    print("WARNING TEST LEAKAGE:", sorted(set(leaks)))
else:
    print("Test leakage check: OK")