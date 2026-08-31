from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = Path(r"C:\Users\Lenovo\Desktop\mynewphotos")
LABEL_DIR = SOURCE_DIR / "yolo_labels"

V5_IMAGE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mixed_all_v5"
    / "images"
    / "train"
)

v5_stems = {p.stem for p in V5_IMAGE_DIR.glob("*.jpg")}

new_items = []

for image in sorted(SOURCE_DIR.glob("*.jpg")):
    stem = image.stem

    # v5 中可能是：
    # 原名
    # new_xxx
    # mouseboost1_new_xxx
    already_in_v5 = any(
        old == stem or old.endswith("_" + stem)
        for old in v5_stems
    )

    if already_in_v5:
        continue

    label = LABEL_DIR / f"{stem}.txt"

    if not label.exists():
        continue

    lines = [
        x.strip()
        for x in label.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    classes = {x.split()[0] for x in lines}

    if not classes:
        kind = "negative"
    elif classes == {"0"}:
        kind = "mouse_only"
    elif classes == {"1"}:
        kind = "phone_only"
    elif classes == {"0", "1"}:
        kind = "both"
    else:
        kind = "other"

    new_items.append((stem, kind))


counts = {
    "mouse_only": 0,
    "phone_only": 0,
    "both": 0,
    "negative": 0,
    "other": 0,
}

for stem, kind in new_items:
    counts[kind] += 1

print("New images since v5:", len(new_items))
print(counts)

print()
for stem, kind in new_items:
    print(kind, stem)