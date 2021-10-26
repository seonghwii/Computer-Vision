import cv2


if __name__ == "__main__":
    src = cv2.imread("picture/Rena.png")
    img = src.copy()
    # 원본 이미지를 훼손시키지 않기 위해 이미지 copy

    print(type(img)) #ndarray => n차원 배열/3차원 => RGB라서 3차원
    # print(img.shape) #(512, 512, 3)


    white = [255, 255, 255]
    blue = [255, 0, 0]
    red = [0, 0, 255]
    green = [0, 255, 0]

    # img에 선을 색깔로 보여준다.
    for i in range(480, 500):
        img[100, i] = white
        #img[y, x] = img[row, col]

    for i in range(100, 301):
        img[100, i], img[i, 100], img[300, i], img[i, 300] = white, red, green, blue

    img[200, :512] = green
    img[:522, 100] = blue

    print(img[100, 100])
    # img[100, 100] = [0, 255, 0]
    """255가 가장 밝은 색/순서 : [B, G, R] || row가 height"""

    # 얼굴 이미지 crop (대략적인 좌표로 찾는다.)
    face_img = img[240:400, 217:375]
    # cv2.imshow("face", face_img)
    """img[y, x].copy() ==> deep copy 
        deep copy를 하지 않는 이유 : 메모리를 덜 사용하기 위해서"""

    # face_img[:, :, 0] = 255 (blue)
    # face_img[:, :, 1] = 255 (Green)
    # face_img[:, :, 2] = 255 (Red)

    # 저장하는 방법("저장할 파일명", 변수명)
    # cv2.imwrite("Rena_face.png", face_img)

    # 채널별로 이미지 확인하는 방법
    b, g, r = cv2.split(img)

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    # cv2.imshow("blue channel", b)
    # cv2.imshow("green channel", g)
    # cv2.imshow("red channel", r)


    # 합친 사진
    merge_img = cv2.merge([r, g, b])
    # cv2.imshow("merge",merge_img)


    # img to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)

    # img to HSV
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print(hsv.shape)

    # gray to color
    # gray_to_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # print(gray_to_color.shape)
    # print(gray_to_color[0, 0])
    """gray를 다시 color로 바꾸려면 평균치를 내서 다 똑같이 적용하므로 조심해서 사용해야 한다."""

    w_width = 200
    w_height = 200

    # 이미지 전체를 2픽셀씩 돌면서 객체를 검출하고자 하는 기술
    for y in range(0, gray.shape[0] - w_height, 2):    #shape[0] = row(height)
        for x in range(0, gray.shape[1] - w_width, 2): #shape[1] = column(width)
            crop = gray[y:y+w_height, x:x + w_width]
            cv2.imshow("crop", crop)
            key = cv2.waitKey(20)

            if key == 27:
                cv2.destroyAllWindows()
                exit(0)

cv2.waitKey(0)
cv2.destroyAllWindows()


