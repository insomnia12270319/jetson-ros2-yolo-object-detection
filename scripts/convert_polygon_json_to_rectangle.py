import json
import shutil
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\Lenovo\Desktop\myphotos")
BACKUP_DIR = SOURCE_DIR / "json_backup"

BACKUP_DIR.mkdir(exist_ok=True)

json_files = list(SOURCE_DIR.glob("*.json"))

converted_count = 0

for json_path in json_files:
    # 先备份原始 JSON
    shutil.copy2(json_path, BACKUP_DIR / json_path.name)

    data = json.loads(json_path.read_text(encoding="utf-8"))

    for shape in data.get("shapes", []):
        points = shape.get("points", [])

        if len(points) < 2:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

        # AnyLabeling rectangle 用两个对角点
        shape["points"] = [
            [x1, y1],
            [x2, y2],
        ]

        shape["shape_type"] = "rectangle"

    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    converted_count += 1

print("Converted JSON files:", converted_count)
print("Backup saved to:", BACKUP_DIR)