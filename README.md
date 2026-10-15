# RecoG

Real-time hand and finger gesture recognition from a webcam, mapped to actions like playing sounds.

> 🚧 Work in progress

## Tech stack
- Python 3.12
- OpenCV — camera capture and display
- MediaPipe — hand landmark detection (21 points per hand)
- pygame — sound playback

## Hardware
- Windows 11
- Logitech C920 webcam

## Setup
```powershell
git clone https://github.com/<your-username>/RecoG.git
cd RecoG
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```powershell
python camera_test.py
```

## Roadmap
- [x] Camera capture
- [ ] Hand landmark tracking
- [ ] Gesture classification
- [ ] Gesture → sound mapping
- [ ] Demo GIF
