# Face Detector Placeholder

The final demo used the OpenFace-supported detection/crop path. This folder is for optional fallback face-detector weights used during configuration and debugging.

Optional YOLOv8 face weights can be placed here:

```text
yolov8n-face.pt
```

If absent, the app falls back to OpenCV Haar detection when OpenCV is installed. The smoke check does not require face detector weights.
