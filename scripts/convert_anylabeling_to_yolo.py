import json
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\Lenovo\Desktop\myphotos")
OUTPUT_DIR = SOURCE_DIR / "yolo_labels"

CLASS_MAP = {
    "mouse": 0,
    "cell phone": 1,
}

OUTPUT_DIR.mkdir(exist_ok=True)


def convert_json(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    yolo_lines = []

    for shape in data.get("shapes", []):
        label = shape["label"]

        if label not in CLASS_MAP:
            print(f"Skip unknown label: {label}")
            continue

        points = shape["points"]

        # polygon / rectangle 都统一取外接矩形
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        box_width = (x2 - x1) / image_width
        box_height = (y2 - y1) / image_height

        class_id = CLASS_MAP[label]

        yolo_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    output_path = OUTPUT_DIR / f"{json_path.stem}.txt"
    output_path.write_text("\n".join(yolo_lines), encoding="utf-8")

    print(f"Converted: {json_path.name}")


json_files = list(SOURCE_DIR.glob("*.json"))

for json_path in json_files:
    convert_json(json_path)

print(f"\nFinished: {len(json_files)} JSON files")
print("YOLO labels saved to:", OUTPUT_DIR)