#from ultralytics import YOLO

#model = YOLO("best.pt")
#model.predict(source="images/", save=True)
#-----------------------------------------------------------------

from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Run prediction on video
#model.predict(
#    source="images/",
#    imgsz=960,      # increase if bottles are small
#    conf=0.15,      # lower = more detections
#    iou=0.6,
#    save=True
#)

model.track(
    source="images/",
    imgsz=960,
    conf=0.25,
    iou=0.6,
    save=True
)