from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

video_file = "videos/Barcelona opera reopens with performance for more than 2000 potted plants.mp4"

results = model(video_file, save=True, conf=0.25)

annotated_frame = results[0].plot()
cv2.imshow("YOLO11 Detection", annotated_frame)

print("done")
