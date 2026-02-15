import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Đọc ảnh gốc ---
image_path = "D:/py/test/test/Receipt-Information-Extraction-main/Receipt-Information-Extraction-main/result/9.jpg"  # <-- thay ảnh gốc ở đây
img = cv2.imread(image_path)

# --- Đọc file tọa độ ---
txt_path = "D:/py/test/test/Receipt-Information-Extraction-main/Receipt-Information-Extraction-main/result/9.txt"  # <-- file chứa dữ liệu bạn đưa ở trên
with open(txt_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split(",")
    if len(parts) < 9:
        continue
    coords = list(map(int, parts[:8]))  # lấy 8 số tọa độ
    transcript = parts[8]

    # Chuyển thành numpy array (4 điểm)
    pts = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))

    # Vẽ polygon
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Vẽ transcript ở góc đầu
    cv2.putText(img, transcript, (coords[0], coords[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# --- Hiển thị bằng matplotlib ---
plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
