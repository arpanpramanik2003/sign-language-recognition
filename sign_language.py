import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r'P:\Documents(p)\Python_Advance\ML Project\Sign_Language\asl_gesture_model.keras')
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
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('ASL Gesture Recognition', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()