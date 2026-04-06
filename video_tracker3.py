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


vehicle_ids = set()
# read video frame by frame
while cap.isOpened():
    success, frame = cap.read()
        
    if not success:
        break 

    # 3. Track vehicles (persist=True is the magic word for IDs)
    results = model.track(source=frame, classes=[2, 3, 5, 7], conf=0.6, persist=True, verbose=False)

    # 4. Extract both IDs and Coordinates for Speed Math
    boxes = results[0].boxes
    if boxes is None:
        boxes = []
    
    frame_height,frame_width = frame.shape[:2]

    left_boundary = (int)(frame_width*0.33) 
    right_boundary = (int)(frame_width*0.66) 

    left_count = 0
    center_count = 0
    right_count = 0


    for box in boxes:
        # Only print data if the model successfully assigned an ID
        if box.id is not None:
            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            cords = [round(x) for x in box.xyxy[0].tolist()]

            vehicle_ids.add(track_id)
            x1, y1, x2, y2 = cords
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # region classification
            if cx < left_boundary:
                left_count += 1
            elif cx < right_boundary:
                center_count += 1
            else:
                right_count += 1
            
            # This is the exact data we need for Part 2!
            #print(f"Vehicle ID: {track_id} | Class: {class_id} | Coordinates: {cords}")

    # Traffic Density
    density = len(boxes) if boxes is not None else 0
    if density < 10:
        level = "LOW"
    elif density < 25:
        level = "MEDIUM"
    else:
        level = "HIGH"

    # Signal Logic
    if density > 25:
        green_time = 60
    elif density > 10:
        green_time = 40
    else:
        green_time = 20

    def get_density(count):
        if count < 5:
            return "LOW"
        elif count < 15:
            return "MEDIUM"
        else:
            return "HIGH"

    left_density = get_density(left_count)
    center_density = get_density(center_count)
    right_density = get_density(right_count)


    # 5. Draw the boxes and write the frame
    annotated_frame = results[0].plot()



    cv2.line(annotated_frame, (left_boundary, 0), (left_boundary, frame_height), (0,255,0), 3)
    cv2.line(annotated_frame, (right_boundary, 0), (right_boundary, frame_height), (0,0,255), 3)


    # print vehicle count
    cv2.putText(annotated_frame, f"Total Vehicles: {len(vehicle_ids)}",
            (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    # displaying density
    cv2.putText(annotated_frame, f"Density: {level}",
            (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    # displaying Signal Logic
    cv2.putText(annotated_frame, f"Green Time: {green_time}s",
            (20,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    # display region density
    cv2.putText(annotated_frame, f"L: {left_density} ({left_count})",
            (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    cv2.putText(annotated_frame, f"C: {center_density} ({center_count})",
                (20,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    cv2.putText(annotated_frame, f"R: {right_density} ({right_count})",
                (20,280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    
    out.write(annotated_frame)

    
    # 6. Show the video
    cv2.imshow("Live Tracking & Saving - Press 'q' to quit", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release() 
cv2.destroyAllWindows()