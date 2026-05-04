# Face Recognition

Simple webcam-based face recognition built with Python, OpenCV, and `face_recognition`.

## Requirements

- Python 3.10+ recommended
- A working webcam
- One or more face images in `images/`

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Start the app from the project root:

```powershell
python main.py
```

Press `Esc` to exit the webcam window.

## Adding Known Faces

Put clear face photos in the `images/` folder. The file name becomes the displayed name, for example:

- `images\Harry Styles.webp`
- `images\Steve Jobs.webp`

Only image files with `.jpg`, `.jpeg`, `.png`, or `.webp` are loaded.