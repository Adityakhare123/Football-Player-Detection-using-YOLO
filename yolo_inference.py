from ultralytics import YOLO

model = YOLO('yolov8l')

result= model.predict('input_videos/08fd33_4.mp4',save=True)
print(result[0])
print("________________________________________________")
for box in result[0].boxes:
    print(box)

