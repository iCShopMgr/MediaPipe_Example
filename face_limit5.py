import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      if len(results.detections) > 5:
          for detection in results.detections:
              mp_drawing.draw_detection(image, detection,
               bbox_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=7))
      else:
          for detection in results.detections:
              mp_drawing.draw_detection(image, detection)

    cv2.imshow('MediaPipe Face Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
