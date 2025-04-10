import cv2
import numpy as np
import tensorflow as tf
import os
import requests

# Dropbox direct download link conversion
dropbox_url = "https://www.dropbox.com/scl/fi/nmktcdiralyyqs8yld9jg/asl_gesture_model.keras?rlkey=c9jdf65bhpqlxfrc61ndo2k44&st=y6xhz6jo&dl=1"
model_path = "asl_gesture_model.keras"

# Download model from Dropbox if not already present
if not os.path.exists(model_path):
    print("Downloading model from Dropbox...")
    response = requests.get(dropbox_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully!")

# Load the trained model
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully")

# Define class labels (29 classes from your ASL dataset)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                'del', 'nothing', 'space']

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess frame (resize to 150x150, normalize)
    img = cv2.resize(frame, (150, 150))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Convert to percentage

    # Display label
    label = f"Gesture: {class_labels[predicted_class]} ({confidence:.2f}%)"
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('ASL Gesture Recognition', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
