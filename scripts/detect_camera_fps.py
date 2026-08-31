from ultralytics import YOLO
import cv2
import time
from pathlib import Path

save_dir = Path(r"C:\Users\Lenovo\Desktop\mouse_hardcases")
save_dir.mkdir(exist_ok=True)

save_count = 0

model = YOLO(
    "runs/detect/mouse_phone_hardcase_v9-2/weights/best.pt"
)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera open failed")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        conf=0.35,
        #关闭检测日志
        verbose=False
    )
  #画框
    annotated = results[0].plot()

    # 计算实时 FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # 显示 FPS
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO Detection", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
     save_count += 1
     save_path = save_dir / f"hard_mouse_{save_count:03d}.jpg"
     cv2.imwrite(str(save_path), frame)
     print("Saved:", save_path)

    elif key == ord("q"):
     break

cap.release()
cv2.destroyAllWindows()