"""
[Face Landmark 검출]
- dlib 사용
"""

import cv2 as cv
import dlib
import numpy as np

if __name__ == '__main__':

    face_detector = dlib.get_frontal_face_detector()
    # HOG : gradient 사용 ==> GRAY 처리
    src = cv.imread("Rena.png")
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    lm_model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    faces = face_detector(gray)

    for face in faces:  # 상 하 : top, bottom | 좌 우 : left, right
        lm = lm_model(src, face)  # face : bounding box (top, bottom, left, right) 위치로 준다.

        lm_point = []
        for p in lm.parts():
            lm_point.append([p.x, p.y])

        lm_point = np.array(lm_point)


        for p in lm_point:
            cv.circle(src, (p[0], p[1]), radius=2, color=(0, 255, 0), thickness=2)

        # 설계도를 참고하여 포인트 확인
        cv.circle(src, (lm_point[36][0], lm_point[36][1]), radius=2, color=(0, 0, 255), thickness=2)
        cv.circle(src, (lm_point[33][0], lm_point[33][1]), radius=2, color=(255, 0, 0), thickness=2)
        cv.circle(src, (lm_point[45][0], lm_point[45][1]), radius=2, color=(0, 0, 0), thickness=2)

        cv.rectangle(src, (face.left(), face.top()), (face.right(), face.bottom()), color=(0, 0, 255), thickness=1)

    cv.imshow("src", src)
    cv.waitKey(0)
    cv.destroyAllWindows()