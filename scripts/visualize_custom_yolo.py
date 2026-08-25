from pathlib import Path
import cv2

stem = "0270629c5484c4d59e9214b076ae4ef6"

image_path = Path(r"C:\Users\Lenovo\Desktop\myphotos") / f"{stem}.jpg"
label_path = Path(r"C:\Users\Lenovo\Desktop\myphotos\yolo_labels") / f"{stem}.txt"
output_path = Path(r"C:\Users\Lenovo\Desktop\myphotos") / f"{stem}_check.jpg"

image = cv2.imread(str(image_path))
h, w = image.shape[:2]

names = {
    0: "mouse",
    1: "cell phone",
}

for line in label_path.read_text().splitlines():
    cls, xc, yc, bw, bh = map(float, line.split())
    cls = int(cls)

    xc *= w
    yc *= h
    bw *= w
    bh *= h

    x1 = int(xc - bw / 2)
    y1 = int(yc - bh / 2)
    x2 = int(xc + bw / 2)
    y2 = int(yc + bh / 2)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        image,
        names[cls],
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

cv2.imwrite(str(output_path), image)
print("Saved:", output_path)