from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("yolov8n.pt")

# Run detection (filtering for Car(2), Bike(3), Bus(5), Truck(7))
results = model.predict(source="traffic.jpg", classes=[2, 3, 5, 7])

# Extract the image with the boxes drawn on it
annotated_image = results[0].plot()

# This saves the picture to your folder and names it 'saved_output.jpg'
cv2.imwrite("saved_output.jpg", annotated_image)

# Use OpenCV to pop open a window showing the result
cv2.imshow("Vehicle Detection Test", annotated_image)

# Wait for you to press any key, then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()