from ultralytics import YOLO
import cv2
import csv
import subprocess
import os
import imageio_ffmpeg


def process_detections(boxes, vehicle_ids, left_boundary, right_boundary):
    left_count = 0
    center_count = 0
    right_count = 0

    for box in boxes:
        if box.id is not None:
            track_id = int(box.id[0])

            vehicle_ids.add(track_id)

            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            cx = int((x1 + x2) / 2)

            if cx < left_boundary:
                left_count += 1
            elif cx < right_boundary:
                center_count += 1
            else:
                right_count += 1

    vehicles_in_frame = len(boxes)
    total_vehicles = len(vehicle_ids)

    return vehicles_in_frame, total_vehicles, left_count, center_count, right_count

def calculate_density(vehicle_count):
    if vehicle_count < 10:
        return "LOW"
    elif vehicle_count < 25:
        return "MEDIUM"
    else:
        return "HIGH"
    
def calculate_signal_time(density):
    if density == "HIGH":
        return 60
    elif density == "MEDIUM":
        return 40
    else:
        return 20
    
def log_to_csv(writer, frame_no, vehicles_in_frame, total_vehicles,
               left_count, center_count, right_count, density, green_time):
    
    writer.writerow([
        frame_no,
        vehicles_in_frame,
        total_vehicles,
        left_count,
        center_count,
        right_count,
        density,
        green_time
    ])

def get_density(count):
    if count < 5:
        return "LOW"
    elif count < 15:
        return "MEDIUM"
    else:
        return "HIGH"

def process_video(video_path):

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    # --- Video Writer Setup ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_output = 'temp_tracked_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    # --------------------------


    vehicle_ids = set()
    frame_no = 0

    csv_file = open('traffic_log.csv', 'w', newline='')
    writer = csv.writer(csv_file)

    writer.writerow([
        "Frame",
        "Vehicles_in_Frame",
        "Total_Vehicles",
        "Left_Count",
        "Center_Count",
        "Right_Count",
        "Density",
        "GreenTime"
    ])

    # read video frame by frame
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        success, frame = cap.read()
            
        if not success or frame is None:
            print("Video ended properly")
            break

        # 3. Track vehicles (persist=True is the magic word for IDs)
        results = model.track(source=frame, classes=[2, 3, 5, 7], conf=0.5, persist=True, verbose=False)

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


        vehicles_in_frame, total_vehicles, left_count, center_count, right_count = \
        process_detections(boxes, vehicle_ids, left_boundary, right_boundary)


        level = calculate_density(vehicles_in_frame)

        green_time = calculate_signal_time(level)

        left_density = get_density(left_count)
        center_density = get_density(center_count)
        right_density = get_density(right_count)

        vehicles_in_frame = len(boxes)
        total_vehicles = len(vehicle_ids)

        frame_no += 1

        log_to_csv(
            writer,
            frame_no,
            vehicles_in_frame,
            total_vehicles,
            left_count,
            center_count,
            right_count,
            level,
            green_time
        )

        csv_file.flush()        #forces immediate data writing in csv file while program is running

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


    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    csv_file.close()

    # Re-encode to H.264 so browsers can play it
    final_output = 'saved_tracked_video.mp4'
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()  # get bundled ffmpeg path
    if os.path.exists(final_output):
        os.remove(final_output)
    subprocess.run([
        ffmpeg_exe, '-y',
        '-i', temp_output,
        '-vcodec', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        final_output
    ], check=True)
    try:
        os.remove(temp_output)  # delete temp file
    except PermissionError:
        pass  # Windows may still hold the file briefly, safe to ignore

    return final_output, "traffic_log.csv"


