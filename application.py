from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import tempfile
import os
import math
import numpy as np  # ✅ Missing import added here

def calculate_angle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(ang) if abs(ang) <= 180 else 360 - abs(ang)

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def home():
    return 'Posture Detection Backend is Running!'

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello from Flask!'})

@app.route('/analyze', methods=['POST'])
def analyze_posture():
    # ✅ Case 1: If video is uploaded
    if 'file' in request.files:
        file = request.files['file']

        # Check if it's a video by extension (quick check)
        filename = file.filename.lower()
        if filename.endswith(('.mp4', '.avi', '.mov')):
            # It's a video file
            temp_video_path = os.path.join(tempfile.gettempdir(), file.filename)
            file.save(temp_video_path)

            cap = cv2.VideoCapture(temp_video_path)
            frame_count = 0
            bad_posture_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

                    a = (left_hip.x, left_hip.y)
                    b = (left_knee.x, left_knee.y)
                    c = (left_ankle.x, left_ankle.y)

                    knee_angle = calculate_angle(a, b, c)
                    back_angle = calculate_angle((left_shoulder.x, left_shoulder.y), (left_hip.x, left_hip.y), (left_ankle.x, left_ankle.y))

                    if left_knee.x > left_ankle.x or back_angle < 150:
                        bad_posture_count += 1

            cap.release()
            os.remove(temp_video_path)

            if bad_posture_count > 0:
                feedback = f'Bad posture detected in {bad_posture_count} frames!'
            else:
                feedback = 'Posture looks good!'

            return jsonify({'result': feedback})

        else:
            # ✅ Case 2: If image is uploaded (webcam capture)
            file_bytes = file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return jsonify({'result': 'Invalid image data.'})

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

                a = (left_hip.x, left_hip.y)
                b = (left_knee.x, left_knee.y)
                c = (left_ankle.x, left_ankle.y)

                knee_angle = calculate_angle(a, b, c)
                back_angle = calculate_angle((left_shoulder.x, left_shoulder.y), (left_hip.x, left_hip.y), (left_ankle.x, left_ankle.y))

                if left_knee.x > left_ankle.x or back_angle < 150:
                    feedback = 'Bad posture detected!'
                else:
                    feedback = 'Posture looks good!'
            else:
                feedback = 'No person detected.'

            return jsonify({'result': feedback})

    else:
        return jsonify({'result': 'No file uploaded.'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
