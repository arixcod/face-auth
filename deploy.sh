#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
download() {
  local url="$1" out="$2"
  if [ ! -s "$out" ]; then
    echo "downloading $out"
    curl -fL --retry 3 -o "$out" "$url"
  fi
}
download https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx models/face_detection_yunet_2023mar.onnx
download https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx models/face_recognition_sface_2021dec.onnx
download https://github.com/facenox/face-antispoof-onnx/releases/download/v1.0.0/best_model.onnx models/minifasnet_v2se.onnx
if ! command -v docker >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-v2
  sudo systemctl enable --now docker
fi
export FACE_API_KEY="${FACE_API_KEY:-change-me}"
sudo FACE_API_KEY="$FACE_API_KEY" docker compose up -d --build
echo "Open http://$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4 || hostname -I | awk '{print $1}'):8080"
