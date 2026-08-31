import argparse
import time

import cv2
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/mouse_phone_yolo26n_hardcase_best.pt",
        help="Path to YOLO model",
    )
    parser.add_argument("--source", default="0", help="Camera index, e.g. 0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    # "0" -> camera 0
    source = int(args.source) if args.source.isdigit() else args.source

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Loading model:", args.model)
    print("Device:", device)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("ERROR: Cannot open camera:", source)
        return

    previous_time = time.perf_counter()

    while True:
        success, frame = cap.read()

        if not success:
            print("ERROR: Cannot read camera frame")
            break

        results = model.predict(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )

        result = results[0]

        # YOLO 自动画 bbox、类别、confidence
        display_frame = result.plot()

        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - previous_time, 1e-6)
        previous_time = current_time

        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Mouse & Cell Phone Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # q 或 ESC 退出
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()