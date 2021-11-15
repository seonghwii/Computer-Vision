import cv2 as cv
import numpy as np
import face_recognition as fr

if __name__ == '__main__':

    src = fr.load_image_file("me.jpg")

    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)

    faces = fr.face_locations(src)
    face_encord = fr.face_encodings(src)[0]

    # data랑 name 지정
    face_db = []
    face_db.append(face_encord)

    face_name = []
    face_name.append("unknown")


    for face in faces:
        top, right, bottom, left = face
        cv.rectangle(src, (left, top), (right, bottom), (0, 0, 255), 3)

    cv.imshow("face", src)

    ####################################################################
    #   새로운 이미지와 비교    #

    input_img = fr.load_image_file("img.jpg")
    input_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
    faces = fr.face_locations(input_img)


    for face in faces:
        top, right, bottom, left = face

        ext_encord = fr.face_encodings(input_img)[0]


        face_dist = fr.face_distance(face_db, ext_encord)  # 원래 점과 지금 뽑은 점들과의 거리
        print("face distance : " , face_dist)
        is_match = fr.compare_faces(face_db, ext_encord)
        best_match = np.argmin(face_dist)
        name = face_name[best_match]
        print("is match : " , is_match)

        cv.rectangle(input_img, (left, top), (right, bottom), (0, 0, 255), 3)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(input_img, name, (left + 6, top + 6), font, 3.0, (255, 255, 0), 2)

    cv.imshow("img", input_img)
    cv.waitKey(0)

    cv.destroyAllWindows()




