from pathlib import Path

import cv2
import face_recognition


class SimpleFacerec:
	def __init__(self):
		self.known_face_encodings = []
		self.known_face_names = []
		self.frame_resizing = 0.25

	def load_encoding_images(self, images_path):
		images_dir = Path(images_path)
		if not images_dir.exists():
			raise FileNotFoundError(f"Image folder not found: {images_dir}")

		for image_file in sorted(images_dir.iterdir()):
			if image_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
				continue

			image = cv2.imread(str(image_file))
			if image is None:
				print(f"Skipping unreadable image: {image_file}")
				continue

			rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			face_locations = face_recognition.face_locations(rgb_image)
			if not face_locations:
				print(f"No face found in: {image_file.name}")
				continue

			face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
			self.known_face_encodings.append(face_encoding)
			self.known_face_names.append(image_file.stem)

		if not self.known_face_encodings:
			raise ValueError(f"No valid face encodings found in {images_dir}")

	def detect_known_faces(self, frame):
		small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
		rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
		face_locations = face_recognition.face_locations(rgb_frame, model="hog")
		face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

		face_names = []
		for face_encoding in face_encodings:
			matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
			name = "Unknown"

			face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
			if len(face_distances) > 0:
				best_match_index = face_distances.argmin()
				if matches[best_match_index]:
					name = self.known_face_names[best_match_index]

			face_names.append(name)

		face_locations = [
			(
				int(top / self.frame_resizing),
				int(right / self.frame_resizing),
				int(bottom / self.frame_resizing),
				int(left / self.frame_resizing),
			)
			for top, right, bottom, left in face_locations
		]

		return face_locations, face_names
