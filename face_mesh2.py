import cv2
import mediapipe as mp
import numpy as np
import statistics
import math

# 開啟標註功能與人臉網格功能
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 紀錄左眼周圍的點
eye_list = [33, 246, 7, 161, 163, 160, 144, 159, 145, 158, 153, 157, 154, 173, 155, 133]

# 設定攝影機
cap = cv2.VideoCapture(0)

# 載入更換眼睛圖案的圖片
eye_normal = cv2.imread("path2880.png")

# 設定檢測率
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        # 從攝影機取得一張畫面
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 轉換顏色為RGB並丟入face mesh運算
        h, w, d = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            # 歷遍左眼各點取得目前座標與眼睛大小
            for face_landmarks in results.multi_face_landmarks:
                eye_point = []
                eye_size = []
                for index in eye_list:
                    x = int(face_landmarks.landmark[index].x * w)
                    y = int(face_landmarks.landmark[index].y * h)
                    eye_point.append([x, y])
                    if index == 153 or index == 159:
                        eye_size.append([x, y])
                if len(eye_size) == 2:
                    eye_len = int(math.pow(math.pow((eye_size[0][0] - eye_size[1][0]), 2) + math.pow((eye_size[0][1] - eye_size[1][1]), 2), 0.5))

            try:
                # 將取得的各點透過statistics計算出眼睛中心座標
                points = eye_point
                center = [statistics.mean(i) for i in zip(*points)]

                # 透過剛剛取得的眼睛大小, 將眼睛圖案的圖片轉換成適合的大小
                eye = cv2.resize(eye_normal, (eye_len-7, eye_len-7))

                # 透過一系列的處理將眼睛圖片貼在左眼上
                eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                _, eye_mask = cv2.threshold(eye_gray, 25, 255, cv2.THRESH_BINARY_INV)

                img_height, img_width, _ = eye.shape
                x, y = int(center[0]-img_width/2), int(center[1]-img_height/2)
                eye_area = frame[y: y+img_height, x: x+img_width]
                eye_area_no_eye = cv2.bitwise_and(eye_area, eye_area, mask=eye_mask)
                final_eye = cv2.add(eye_area_no_eye, eye)
                frame[y: y+img_height, x: x+img_width] = final_eye

            except:
                pass

        cv2.imshow("MediaPipe FaceMesh", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
