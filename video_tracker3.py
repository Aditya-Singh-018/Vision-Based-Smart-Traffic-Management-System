from ultralytics import YOLO
import cv2

# 1. Load the model
model = YOLO("yolov8n.pt")

# 2. Open the video file
video_path = "traffic_video.mp4"
cap = cv2.VideoCapture(video_path)
# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# --- Video Writer Setup ---
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Fix for FPS issue
if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('saved_tracked_video.mp4', fourcc, fps, (width, height))
# --------------------------

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break 

    # 3. Track vehicles (persist=True is the magic word for IDs)
    results = model.track(source=frame, classes=[2, 3, 5, 7], conf=0.6, persist=True, verbose=False)

    # 4. Extract both IDs and Coordinates for Speed Math
    boxes = results[0].boxes
    
    for box in boxes:
        # Only print data if the model successfully assigned an ID
        if box.id is not None:
            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            cords = [round(x) for x in box.xyxy[0].tolist()]
            
            # This is the exact data we need for Part 2!
            print(f"Vehicle ID: {track_id} | Class: {class_id} | Coordinates: {cords}")

    # 5. Draw the boxes and write the frame
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    # 6. Show the video
    cv2.imshow("Live Tracking & Saving - Press 'q' to quit", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release() 
cv2.destroyAllWindows()