import pandas as pd
import cv2 as cv
import mediapipe as mp
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class FireController:
    def __init__(self):
        self.initialized = True
        self.dataset_path = Path(__file__).parent / "fires_dataset.csv"
        self.model_filename = Path(__file__).parent / "fire_controller.joblib"
        self.model = None

        # Create models directory if it doesn't exist
        self.model_filename.parent.mkdir(parents=True, exist_ok=True)
        
        self.load_model()

    def decide(self, points):
        """Predict whether to fire based on hand points"""
        if self.model is None:
            print("Warning: Model not loaded. Training first...")
            self.train()
        
        # Convert points to the format expected by the model
        points_array = np.array(points).reshape(1, -1)
        prediction = self.model.predict(points_array)
        
        return bool(prediction[0])

    def load_model(self):
        """Load the trained model from file"""
        try:
            if self.model_filename.exists():
                self.model = joblib.load(self.model_filename)
                print(f"Model loaded successfully from {self.model_filename}")
            else:
                print(f"Model file not found at {self.model_filename}")
                print("Please train the model first using fire_controller.train()")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def save_model(self, model):
        """Save the trained model to file"""
        try:
            joblib.dump(model, self.model_filename)
            print(f"Model saved successfully to {self.model_filename}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def train(self):
        """Train the fire detection model"""
        try:
            df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            print("Error: fires_dataset.csv not found. Please collect data first.")
            return

        # Preprocessing
        y = df["is_fire"]
        X = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0", "is_fire"])
        X_train,  X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # Define Model
        model = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=10)
        
        # Training
        model.fit(X_train, y_train)

        # Metrics
        y_pred = model.predict(X_test)
        model_accuracy_score = accuracy_score(y_test, y_pred)  # Fixed parameter order
        print(f"Model Accuracy: {model_accuracy_score:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))  # Fixed parameter order
        print("Classification Report:")
        print(classification_report(y_test, y_pred))  # Fixed parameter order

        if model_accuracy_score > 0.90:
            self.save_model(model)
            self.model = model
            print("Model training completed and saved!")
        else:
            print(f"Model accuracy ({model_accuracy_score:.4f}) below threshold (0.90). Model not saved.")

    def collect_data(self):  # Fixed typo in method name
        """Run Camera and create a Dataset of thumb finger pose"""
        fallback_df: pd.DataFrame = pd.DataFrame()

        try:
            df = pd.read_csv('fires_dataset.csv')
            print("CSV file loaded successfully")
        except:
            df = pd.DataFrame()  # Fixed undefined variable

        # initialize mediapipe hands
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            static_image_mode=False,
        )

        self.cap = cv.VideoCapture(0)
        _, frame = self.cap.read()
    
        self.initial_blank = np.zeros(frame.shape, dtype=np.uint8)
        
        if not self.cap.isOpened():
            print("ERROR: Could not open Camera")
            return

        self.printed_thumb = False

        while True:
            row: pd.DataFrame = pd.DataFrame({
                "is_fire": 1
            }, index=[0])

            ret, frame = self.cap.read()
            blank = self.initial_blank.copy()

            if not ret:
                print("ERROR: Failed to capture frame")
                return 

            # Flip the frame horizontally for mirror effect
            frame = cv.flip(frame, 1)

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # process hands
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
                h, w, c = blank.shape

                first_hand = hand_results.multi_hand_landmarks[0]
                second_hand = hand_results.multi_hand_landmarks[1]
                f_landmark = first_hand.landmark
                l_landmark = second_hand.landmark

                f_thumb = f_landmark[1:5]
                l_thumb = l_landmark[1:5]
                
                for idx, pos in enumerate(f_thumb):
                    tup = (pos.x, pos.y, pos.z)
                    row[f"first-hand-pt-{idx + 1}-x"] = tup[0]
                    row[f"first-hand-pt-{idx + 1}-y"] = tup[1]
                    row[f"first-hand-pt-{idx + 1}-z"] = tup[2]
                
                for idx, pos in enumerate(l_thumb):
                    tup = (pos.x, pos.y, pos.z)
                    row[f"second-hand-pt-{idx + 1}-x"] = tup[0]
                    row[f"second-hand-pt-{idx + 1}-y"] = tup[1]
                    row[f"second-hand-pt-{idx + 1}-z"] = tup[2]

                df = pd.concat([df, row], ignore_index=True)

                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Draw hand landmarks and connections
                    mp_drawing.draw_landmarks(
                        blank,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    cv.imshow("Hand Tracking", blank)
                    cv.imshow("Camera Feed", frame)
            
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
            
        self.cap.release()
        cv.destroyAllWindows()
        self.hands.close()
        print("Camera released and windows closed.")
        print("Saving collected dataset...")
        df.to_csv("fires_dataset.csv")
        print("Dataset Saved Successfully!")


if __name__ == "__main__":
    fire_controller = FireController()
    
    # Example usage:
    # fire_controller.collect_data()  # Collect training data
    # fire_controller.train()         # Train the model
    # result = fire_controller.decide([...])  # Use the model