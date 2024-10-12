import cv2
import numpy as np
from fer import FER

# Initialize emotion detector
detector = FER()

def initialize_video_capture():
    """Initialize video capture from the camera."""
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    if not cap.isOpened():
        raise Exception("Error: Could not open video capture.")
    return cap

def detect_emotions(frame):
    """Detect emotions in a given frame."""
    try:
        emotions = detector.detect_emotions(frame)
        return emotions
    except Exception as e:
        print(f"Error in detecting emotions: {e}")
        return []

def draw_emotion_labels(frame, emotions):
    """Draw the emotion labels on the frame."""
    if emotions:
        # Get the emotion with the highest score
        emotion = emotions[0]
        emotion_text = emotion['emotions']
        dominant_emotion = max(emotion_text, key=emotion_text.get)  # Get the dominant emotion
        accuracy = emotion_text[dominant_emotion]  # Get the accuracy of the dominant emotion

        # Draw the emotion box
        x, y, w, h = emotion['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{dominant_emotion} ({accuracy * 100:.2f}%)", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return dominant_emotion, accuracy  # Return the dominant emotion and its accuracy
    return None, None  # Return None if no emotions are detected

def main():
    """Main function to capture video and detect emotions."""
    try:
        cap = initialize_video_capture()
        screen_width = 1920  # Adjust this according to screen width
        screen_height = 1080  # Adjust this according to  screen height

        final_emotion = None
        final_accuracy = None

        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Error: Could not read frame.")

            # Resize frame to full screen dimensions
            frame = cv2.resize(frame, (screen_width, screen_height))

            emotions = detect_emotions(frame)
            emotion, accuracy = draw_emotion_labels(frame, emotions)

            # Store the final emotion and accuracy
            if emotion and accuracy is not None:
                final_emotion = emotion
                final_accuracy = accuracy

            cv2.imshow('Video Feed', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Print the final detected emotion and accuracy before closing
        if final_emotion and final_accuracy is not None:
            print(f"Final Detected Emotion: {final_emotion} (Accuracy: {final_accuracy * 100:.2f}%)")
        
        cap.release()  # Release the video capture object
        cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
