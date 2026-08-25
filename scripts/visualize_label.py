from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "dataset" / "images" / "train" / "000000000384.jpg"
LABEL_PATH = PROJECT_ROOT / "dataset" / "labels" / "train" / "000000000384.txt"

image = cv2.imread(str(IMAGE_PATH))
h, w = image.shape[:2]

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()

        class_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        box_w = float(parts[3]) * w
        box_h = float(parts[4]) * h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"class {class_id}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

OUTPUT_PATH = PROJECT_ROOT / "dataset" / "coco_raw" / "visualized_000000000384.jpg"
cv2.imwrite(str(OUTPUT_PATH), image)

print("Saved to:", OUTPUT_PATH)