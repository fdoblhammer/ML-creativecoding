from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_file = "videos/Barcelona opera reopens with performance for more than 2000 potted plants.mp4"

results = model(video_file, save=True, conf=0.25)

print("done")
