#!/bin/bash
# Five hand photos from the MediaPipe test assets (Apache-2.0 repo) so you have something to run.
mkdir -p images
for f in thumb_up pointing_up pointing_up_rotated victory fist; do
  curl -sSL -o images/$f.jpg https://storage.googleapis.com/mediapipe-assets/$f.jpg
done
ls -l images
