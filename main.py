from pathlib import Path

import cv2
from simple_facerec import SimpleFacerec

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"

def main():
    sfr = SimpleFacerec()
    print(f"Loading face encodings from {IMAGES_DIR} ...")
    sfr.load_encoding_images(IMAGES_DIR)
    print("Face encodings loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Could not read frame from webcam")
                break

            face_locations, face_names = sfr.detect_known_faces(frame)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()