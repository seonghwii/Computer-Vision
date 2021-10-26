"""
당근 부분만 남도록 이미지를 이진화한 예제
"""
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #실습
    c_b_img = cv2.imread("../picture/c_b.png")
    grey = cv2.cvtColor(c_b_img, cv2.COLOR_BGR2GRAY)

    # norm_img = cv2.normalize(c_b_img_grey, None, 0, 255, cv2.NORM_MINMAX)
    # eq_img = cv2.equalizeHist(norm_img)

    #b, g, r

    g_grey = c_b_img[:, :, 1]
    r_grey = c_b_img[:, :, 2]

    # cv2.imshow("green_img", green_img)
    # cv2.imshow("red_img", red_img)
    # cv2.imshow("blue_img", blue_img)

    #바이너리 이미지로 변환시켜준다._threshold함수는 변수를 주어야 한다.
    r_thr, r_bin_img = cv2.threshold(r_grey, 0, 255, cv2.THRESH_OTSU)
    print(r_thr) #threshold를 알 수 있음.

    g_thr, g_bin_img = cv2.threshold(g_grey, 110, 255, cv2.THRESH_OTSU)
    print(g_thr)  # threshold를 알 수 있음.


    #바이너리 이미지를 bgr컬러로 바꾸어준다.
    g_bin_img = cv2.cvtColor(g_bin_img, cv2.COLOR_GRAY2BGR)
    r_bin_img = cv2.cvtColor(r_bin_img, cv2.COLOR_GRAY2BGR)

    #mask 이미지로 씌운다.
    mask_img = cv2.bitwise_and(c_b_img, r_bin_img)
    mask_img = cv2.bitwise_and(mask_img, g_bin_img)

    cv2.imshow("masked", mask_img)
    cv2.imshow("r_bin_img", r_bin_img)
    cv2.imshow("g_bin_img", g_bin_img)
    cv2.imshow("c_b_img", c_b_img)

    """
    mask_img의 원리
    mask_img = c_b_img.copy()
    
    for y in range(bin_img.shape[0]):
        for x in range(bin_img.shape[1]):
            if bin_img[y, x] == 0:
                mask_img[y, x] = [0, 0, 0]
    """


    threshold = 100

    thr_1, c_b_img_0 = cv2.threshold(c_b_img, threshold, 255, cv2.THRESH_BINARY)
    thr_inv_1, c_b_img_1 = cv2.threshold(c_b_img, threshold, 255, cv2.THRESH_BINARY_INV)
    thr_trc_1, c_b_img_2 = cv2.threshold(c_b_img, threshold, 255, cv2.THRESH_TRUNC)
    thr_tzr_1, c_b_img_3 = cv2.threshold(c_b_img, threshold, 255, cv2.THRESH_TOZERO)
    thr_tzr_inv_1, c_b_img_4 = cv2.threshold(c_b_img, threshold, 255, cv2.THRESH_TOZERO_INV)

    #threshold 함수의 return 값은 2개이다.


    # 가시화
    # cv2.imshow("img", img)

    # cv2.imshow("bin_img", bin_img_0)
    # cv2.imshow("bin_img_inv", bin_img_1)
    # cv2.imshow("bin_img_trunc", bin_img_2)
    # cv2.imshow("img_tozero", bin_img_3)
    # cv2.imshow("img_tozero_inv", bin_img_4)

    # cv2.imshow("bin_img_inv", c_b_img_1)
    # cv2.imshow("bin_img_trunc", c_b_img_2)
    # cv2.imshow("img_tozero", c_b_img_3)
    # cv2.imshow("img_tozero_inv", c_b_img_4)
    # cv2.imshow("bin_img", c_b_img_0)


    # cv2.imshow("img_norm", img_norm)
    # cv2.imshow("img_eq", img_eq)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()