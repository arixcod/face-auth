# Face register and verify

Models packed in `models/`:

- YuNet `face_detection_yunet_2023mar.onnx` finds one face
- MiniFASNetV2-SE `minifasnet_v2se.onnx` scores live vs photo/screen
- SFace `face_recognition_sface_2021dec.onnx` builds the embedding stored in SQLite

No locker and no booking. Register returns `face_id`. Verify compares a new photo to that id only.

## EC2

```bash
cd face-auth-service
chmod +x deploy.sh
FACE_API_KEY='pick-a-long-key' ./deploy.sh
```

Open `http://<ec2-public-ip>:8080`. Security group: TCP 8080 from your IP.

Match cutoff is cosine `0.363` (SFace default). Liveness must be at least `0`.
