#!/usr/bin/env bash
# Run format: bash -i data-pipeline <Video folder> <Script folder> <Audio filename for lipgan>
# Script will:
# 1. Split video into frames
# 2. Crop around the faces in these frames, and save the bounding box coordinates to a txt file
# 3. Run 2D landmark detection on these cropped images
# 4. Save a visual of the results
# 5. Deepfake creation, with steps 1-3 repeated
#
# Assumes 6 cameras with 29.9fps mp4 video, named camera1.MP4, ..., camera6.MP4
#
# Written by Eleanor Tursman
# Last updated 7/2020

VIDEO_LOCATION=${1:-""}
SCRIPT_LOCATION=${2:-""}
AUDIO_FILENAME=${3:-""}

# Exit the moment a command fails
set -euo pipefail

cd "${VIDEO_LOCATION}"

################## Convert video to frames ##################
echo "Converting video to frames..."

mkdir -p "${VIDEO_LOCATION}cam1-frames"
mkdir -p "${VIDEO_LOCATION}cam2-frames"
mkdir -p "${VIDEO_LOCATION}cam3-frames"
mkdir -p "${VIDEO_LOCATION}cam4-frames"
mkdir -p "${VIDEO_LOCATION}cam5-frames"
mkdir -p "${VIDEO_LOCATION}cam6-frames"

ffmpeg -i "camera1.MP4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam1-frames/frames%04d.jpg" 
echo "Video 1 done."
ffmpeg -i "camera2.MP4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam2-frames/frames%04d.jpg" 
echo "Video 2 done."
ffmpeg -i "camera3.MP4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam3-frames/frames%04d.jpg" 
echo "Video 3 done."
ffmpeg -i "camera4.MP4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam4-frames/frames%04d.jpg" 
echo "Video 4 done."
ffmpeg -i "camera5.MP4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam5-frames/frames%04d.jpg" 
echo "Video 5 done."
ffmpeg -i "camera6.MP4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam6-frames/frames%04d.jpg" 
echo "Video 6 done."

################## Get face bounding boxes ##################
echo "Cropping faces..."

mkdir -p "${VIDEO_LOCATION}cam1-cropped"
mkdir -p "${VIDEO_LOCATION}cam2-cropped"
mkdir -p "${VIDEO_LOCATION}cam3-cropped"
mkdir -p "${VIDEO_LOCATION}cam4-cropped"
mkdir -p "${VIDEO_LOCATION}cam5-cropped"
mkdir -p "${VIDEO_LOCATION}cam6-cropped"
mkdir -p "${VIDEO_LOCATION}/bounding-boxes"

python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam1-frames/" "${VIDEO_LOCATION}cam1-cropped/" "${VIDEO_LOCATION}bounding-boxes/cam1-"
echo "Video 1 done."
python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam2-frames/" "${VIDEO_LOCATION}cam2-cropped/" "${VIDEO_LOCATION}bounding-boxes/cam2-"
echo "Video 2 done."
python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam3-frames/" "${VIDEO_LOCATION}cam3-cropped/" "${VIDEO_LOCATION}bounding-boxes/cam3-"
echo "Video 3 done."
python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam4-frames/" "${VIDEO_LOCATION}cam4-cropped/" "${VIDEO_LOCATION}bounding-boxes/cam4-"
echo "Video 4 done."
python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam5-frames/" "${VIDEO_LOCATION}cam5-cropped/" "${VIDEO_LOCATION}bounding-boxes/cam5-"
echo "Video 5 done."
python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam6-frames/" "${VIDEO_LOCATION}cam6-cropped/" "${VIDEO_LOCATION}bounding-boxes/cam6-"
echo "Video 6 done."
#
# ################## Get 2D landmarks ##################
echo "Running 2D landmark detection..."

mkdir -p "${VIDEO_LOCATION}cam1-landmarks"
mkdir -p "${VIDEO_LOCATION}cam2-landmarks"
mkdir -p "${VIDEO_LOCATION}cam3-landmarks"
mkdir -p "${VIDEO_LOCATION}cam4-landmarks"
mkdir -p "${VIDEO_LOCATION}cam5-landmarks"
mkdir -p "${VIDEO_LOCATION}cam6-landmarks"

python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam1-frames/" "${VIDEO_LOCATION}cam1-landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam1-bounding-boxes.txt"
echo "Video 1 done."
python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam2-frames/" "${VIDEO_LOCATION}cam2-landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam2-bounding-boxes.txt"
echo "Video 2 done."
python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam3-frames/" "${VIDEO_LOCATION}cam3-landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam3-bounding-boxes.txt"
echo "Video 3 done."
python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam4-frames/" "${VIDEO_LOCATION}cam4-landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam4-bounding-boxes.txt"
echo "Video 4 done."
python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam5-frames/" "${VIDEO_LOCATION}cam5-landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam5-bounding-boxes.txt"
echo "Video 5 done."
python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam6-frames/" "${VIDEO_LOCATION}cam6-landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam6-bounding-boxes.txt"
echo "Video 6 done."

################## Visual of 2D landmarks on original input ##################
echo "Creating visual of results..."

mkdir -p "${VIDEO_LOCATION}/visualization"

python3 "${SCRIPT_LOCATION}visualize-landmarks.py" "${VIDEO_LOCATION}"
ffmpeg -i "${VIDEO_LOCATION}visualization/%04d.jpg" -loglevel quiet -vf scale=1366:768 -q:v 2 "${VIDEO_LOCATION}landmarks-visual.mp4"

################## Create deepfake ##################
echo "Creating deepfake with LipGan..."

mkdir -p "${VIDEO_LOCATION}cam1-lipgan"
mkdir -p "${VIDEO_LOCATION}cam2-lipgan"
mkdir -p "${VIDEO_LOCATION}cam3-lipgan"
mkdir -p "${VIDEO_LOCATION}cam4-lipgan"
mkdir -p "${VIDEO_LOCATION}cam5-lipgan"
mkdir -p "${VIDEO_LOCATION}cam6-lipgan"

# To run lipgan on our machine, we needed to downsample & restrict the video length to the first 4000 frames
ffmpeg -i "${VIDEO_LOCATION}camera1.MP4" -ss 0.0 -frames:v 4000 -framerate 29.97 -vf scale=1280:720 -crf 0 "${VIDEO_LOCATION}cam1-lipgan/cam1-first-4k.mp4"
echo "Video 1 done"
ffmpeg -i "${VIDEO_LOCATION}camera2.MP4" -ss 0.0 -frames:v 4000 -framerate 29.97 -vf scale=1280:720 -crf 0 "${VIDEO_LOCATION}cam2-lipgan/cam2-first-4k.mp4"
echo "Video 2 done"
ffmpeg -i "${VIDEO_LOCATION}camera3.MP4" -ss 0.0 -frames:v 4000 -framerate 29.97 -vf scale=1280:720 -crf 0 "${VIDEO_LOCATION}cam3-lipgan/cam3-first-4k.mp4"
echo "Video 3 done"
ffmpeg -i "${VIDEO_LOCATION}camera4.MP4" -ss 0.0 -frames:v 4000 -framerate 29.97 -vf scale=1280:720 -crf 0 "${VIDEO_LOCATION}cam3-lipgan/cam4-first-4k.mp4"
echo "Video 4 done"
ffmpeg -i "${VIDEO_LOCATION}camera5.MP4" -ss 0.0 -frames:v 4000 -framerate 29.97 -vf scale=1280:720 -crf 0 "${VIDEO_LOCATION}cam5-lipgan/cam5-first-4k.mp4"
echo "Video 5 done"
ffmpeg -i "${VIDEO_LOCATION}camera6.MP4" -ss 0.0 -frames:v 4000 -framerate 29.97 -vf scale=1280:720 -crf 0 "${VIDEO_LOCATION}cam6-lipgan/cam6-first-4k.mp4"
echo "Video 6 done"

# Create new bboxes based on landmark fit
python "${SCRIPT_LOCATION}get-new-bboxes.py" 1 "${VIDEO_LOCATION}"
python "${SCRIPT_LOCATION}get-new-bboxes.py" 2 "${VIDEO_LOCATION}"
python "${SCRIPT_LOCATION}get-new-bboxes.py" 3 "${VIDEO_LOCATION}"
python "${SCRIPT_LOCATION}get-new-bboxes.py" 4 "${VIDEO_LOCATION}"
python "${SCRIPT_LOCATION}get-new-bboxes.py" 5 "${VIDEO_LOCATION}"
python "${SCRIPT_LOCATION}get-new-bboxes.py" 6 "${VIDEO_LOCATION}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lipgan

cd "${SCRIPT_LOCATION}../LipGAN/"

python batch_inference.py --checkpoint_path logs/lipgan_best_residual.h5 --face "${VIDEO_LOCATION}cam1-lipgan/cam1-first-4k.mp4" --fps 29.97 --audio "audio/${AUDIO_FILENAME}.wav" --mat "audio/${AUDIO_FILENAME}.mat" --results_dir results/ --bboxes "${VIDEO_LOCATION}/bounding-boxes/cam1-lipgan-bounding-boxes.txt"
mv results/result_voice.mp4 "${VIDEO_LOCATION}cam1-lipgan/cam1-lipgan.mp4"

python batch_inference.py --checkpoint_path logs/lipgan_best_residual.h5 --face "${VIDEO_LOCATION}cam2-lipgan/cam2-first-4k.mp4" --fps 29.97 --audio "audio/${AUDIO_FILENAME}.wav" --mat "audio/${AUDIO_FILENAME}.mat" --results_dir results/ --bboxes "${VIDEO_LOCATION}/bounding-boxes/cam2-lipgan-bounding-boxes.txt"
mv results/result_voice.mp4 "${VIDEO_LOCATION}cam2-lipgan/cam2-lipgan.mp4"

python batch_inference.py --checkpoint_path logs/lipgan_best_residual.h5 --face "${VIDEO_LOCATION}cam3-lipgan/cam3-first-4k.mp4" --fps 29.97 --audio "audio/${AUDIO_FILENAME}.wav" --mat "audio/${AUDIO_FILENAME}.mat" --results_dir results/ --bboxes "${VIDEO_LOCATION}/bounding-boxes/cam3-lipgan-bounding-boxes.txt"
mv results/result_voice.mp4 "${VIDEO_LOCATION}cam3-lipgan/cam3-lipgan.mp4"

python batch_inference.py --checkpoint_path logs/lipgan_best_residual.h5 --face "${VIDEO_LOCATION}cam4-lipgan/cam4-first-4k.mp4" --fps 29.97 --audio "audio/${AUDIO_FILENAME}.wav" --mat "audio/${AUDIO_FILENAME}.mat" --results_dir results/ --bboxes "${VIDEO_LOCATION}/bounding-boxes/cam4-lipgan-bounding-boxes.txt"
mv results/result_voice.mp4 "${VIDEO_LOCATION}cam4-lipgan/cam4-lipgan.mp4"

python batch_inference.py --checkpoint_path logs/lipgan_best_residual.h5 --face "${VIDEO_LOCATION}cam5-lipgan/cam5-first-4k.mp4" --fps 29.97 --audio "audio/${AUDIO_FILENAME}.wav" --mat "audio/${AUDIO_FILENAME}.mat" --results_dir results/ --bboxes "${VIDEO_LOCATION}/bounding-boxes/cam5-lipgan-bounding-boxes.txt"
mv results/result_voice.mp4 "${VIDEO_LOCATION}cam5-lipgan/cam5-lipgan.mp4"

python batch_inference.py --checkpoint_path logs/lipgan_best_residual.h5 --face "${VIDEO_LOCATION}cam6-lipgan/cam6-first-4k.mp4" --fps 29.97 --audio "audio/${AUDIO_FILENAME}.wav" --mat "audio/${AUDIO_FILENAME}.mat" --results_dir results/ --bboxes "${VIDEO_LOCATION}/bounding-boxes/cam6-lipgan-bounding-boxes.txt"
mv results/result_voice.mp4 "${VIDEO_LOCATION}cam6-lipgan/cam6-lipgan.mp4"

conda deactivate

################## Process deepfake ##################
echo "Converting deepfake video to frames..."

mkdir -p "${VIDEO_LOCATION}cam1-lipgan/frames"
mkdir -p "${VIDEO_LOCATION}cam2-lipgan/frames"
mkdir -p "${VIDEO_LOCATION}cam3-lipgan/frames"
mkdir -p "${VIDEO_LOCATION}cam4-lipgan/frames"
mkdir -p "${VIDEO_LOCATION}cam5-lipgan/frames"
mkdir -p "${VIDEO_LOCATION}cam6-lipgan/frames"

ffmpeg -i "${VIDEO_LOCATION}cam1-lipgan/cam1-lipgan.mp4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam1-lipgan/frames/frames%04d.jpg" 
echo "Video 1 done."
ffmpeg -i "${VIDEO_LOCATION}cam2-lipgan/cam2-lipgan.mp4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam2-lipgan/frames/frames%04d.jpg" 
echo "Video 2 done."
ffmpeg -i "${VIDEO_LOCATION}cam3-lipgan/cam3-lipgan.mp4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam3-lipgan/frames/frames%04d.jpg" 
echo "Video 3 done."
ffmpeg -i "${VIDEO_LOCATION}cam4-lipgan/cam4-lipgan.mp4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam4-lipgan/frames/frames%04d.jpg" 
echo "Video 4 done."
ffmpeg -i "${VIDEO_LOCATION}cam5-lipgan/cam5-lipgan.mp4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam5-lipgan/frames/frames%04d.jpg" 
echo "Video 5 done."
ffmpeg -i "${VIDEO_LOCATION}cam6-lipgan/cam6-lipgan.mp4" -loglevel quiet -q:v 2 "${VIDEO_LOCATION}cam6-lipgan/frames/frames%04d.jpg" 
echo "Video 6 done."

echo "Cropping faces..."

mkdir -p "${VIDEO_LOCATION}cam1-lipgan/cropped"
mkdir -p "${VIDEO_LOCATION}cam2-lipgan/cropped"
mkdir -p "${VIDEO_LOCATION}cam3-lipgan/cropped"
mkdir -p "${VIDEO_LOCATION}cam4-lipgan/cropped"
mkdir -p "${VIDEO_LOCATION}cam6-lipgan/cropped"

python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam1-lipgan/frames/" "${VIDEO_LOCATION}cam1-lipgan/cropped/" "${VIDEO_LOCATION}bounding-boxes/cam1-post-lipgan-"
echo "Video 1 done."
python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam2-lipgan/frames/" "${VIDEO_LOCATION}cam2-lipgan/cropped/" "${VIDEO_LOCATION}bounding-boxes/cam2-post-lipgan-"
echo "Video 2 done."
python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam3-lipgan/frames/" "${VIDEO_LOCATION}cam3-lipgan/cropped/" "${VIDEO_LOCATION}bounding-boxes/cam3-post-lipgan-"
echo "Video 3 done."
python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam4-lipgan/frames/" "${VIDEO_LOCATION}cam4-lipgan/cropped/" "${VIDEO_LOCATION}bounding-boxes/cam4-post-lipgan-"
echo "Video 4 done."
python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam5-lipgan/frames/" "${VIDEO_LOCATION}cam5-lipgan/cropped/" "${VIDEO_LOCATION}bounding-boxes/cam5-post-lipgan-"
echo "Video 5 done."
python "${SCRIPT_LOCATION}cnn_face_detector.py" "${SCRIPT_LOCATION}mmod_human_face_detector.dat" "${VIDEO_LOCATION}cam6-lipgan/frames/" "${VIDEO_LOCATION}cam6-lipgan/cropped/" "${VIDEO_LOCATION}bounding-boxes/cam6-post-lipgan-"
echo "Video 6 done."

echo "Running 2D landmark detection..."

mkdir -p "${VIDEO_LOCATION}cam1-lipgan/landmarks"
mkdir -p "${VIDEO_LOCATION}cam2-lipgan/landmarks"
mkdir -p "${VIDEO_LOCATION}cam3-lipgan/landmarks"
mkdir -p "${VIDEO_LOCATION}cam4-lipgan/landmarks"
mkdir -p "${VIDEO_LOCATION}cam5-lipgan/landmarks"
mkdir -p "${VIDEO_LOCATION}cam6-lipgan/landmarks"

python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam1-lipgan/frames/" "${VIDEO_LOCATION}cam1-lipgan/landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam1-post-lipgan-bounding-boxes.txt"
echo "Video 1 done."
python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam2-lipgan/frames/" "${VIDEO_LOCATION}cam2-lipgan/landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam2-post-lipgan-bounding-boxes.txt"
echo "Video 2 done."
python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam3-lipgan/frames/" "${VIDEO_LOCATION}cam3-lipgan/landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam3-post-lipgan-bounding-boxes.txt"
echo "Video 3 done."
python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam4-lipgan/frames/" "${VIDEO_LOCATION}cam4-lipgan/landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam4-post-lipgan-bounding-boxes.txt"
echo "Video 4 done."
python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam5-lipgan/frames/" "${VIDEO_LOCATION}cam5-lipgan/landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam5-post-lipgan-bounding-boxes.txt"
echo "Video 5 done."
python3 "${SCRIPT_LOCATION}detectFeatures.py" "${VIDEO_LOCATION}cam6-lipgan/frames/" "${VIDEO_LOCATION}cam6-lipgan/landmarks/" "${VIDEO_LOCATION}bounding-boxes/cam6-post-lipgan-bounding-boxes.txt"
echo "Video 6 done."








