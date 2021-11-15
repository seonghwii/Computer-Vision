import cv2 as cv
import os


if __name__ == '__main__':
    model_file = "haarcascade_frontalface_alt.xml"
    clf = cv.CascadeClassifier(model_file)

    src = cv.imread("Rena.png")
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # detectMultiScale ==> rectangle의 list가 나온다.
    result = clf.detectMultiScale(gray, # 입력 이미지
                                  scaleFactor=1.2, # 이미지 피라미드 스케일 factor
                                  minNeighbors=1, # 인접 객체 최소 거리 픽셀
                                  minSize=(25, 25)) # 탐지 객체 최소 크기

    for box in result:
        x, y, w, h = box
        sub_img = src[y:y+w, x:x+w]
        cv.rectangle(src, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)


    # 사진 출력
    cv.imshow("iu", src)
    cv.imshow("sub", sub_img)
    # cv.imwrite()
    cv.waitKey(0)
    cv.destroyAllWindows()

"""
[참고사항]
for box in result:
        x, y, w, h = box
        sub_img = src[y:y+w, x:x+w]
        # cv.rectangle(src, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)   
        # src 에서 x, y 에서부터 x+w, y+h 까지
        cv.imshow("sub", sub_img)
        cv.waitKey(0)
    cv.imwrite('1.jpg', sub_img)

    cv.imshow("src", src)
    cv.waitKey(0)
    cv.destroyAllWindows()
"""