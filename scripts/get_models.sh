#!/bin/bash
# Download the two MediaPipe hand models (the .task file is a zip holding both .tflite files).
set -e
mkdir -p palm_det hand_lm; cd palm_det
curl -sSL -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
python3 -c "import zipfile;zipfile.ZipFile('hand_landmarker.task').extractall('.')"
mv hand_landmarks_detector.tflite ../hand_lm/; cd ..; ls -l palm_det hand_lm
