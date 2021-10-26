"""
[히스토그램 구현]
명암 값이 각각 영상에 몇 번 나타나는지 가시적으로 나타내기 위해 히스토그램을 사용한다.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    #사진 가져오기
    img = cv2.imread("img2.jpg", cv2.IMREAD_GRAYSCALE)
    gray = img.copy()

    #numpy library 사용한 histogram 구현
    hist_src = np.bincount(img.ravel(), minlength=256)
    """bincount는 1차원만 받기 때문에 ravel()을 사용하여 1차원으로 변경시켜준다."""

    # openCV, numpy를 이용한 normalization 구현
    """NORM_MINMAX : 다른 값들을 0(최솟값)과 1(최댓값) 사이의 값으로 변환하여 normalization 해주는 기능"""
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    print(img_norm)
    hist_norm = np.bincount(img_norm.ravel(), minlength=256)

    #equalization
    img_eq = cv2.equalizeHist(img_norm)
    hist_eq = np.bincount(img_eq.ravel(), minlength=256)


    #subplot : 이미지를 여러 개 띄우고 싶을 때 (행, 열, 이미지 순서)
    #plot : 보여주는 기능
    plt.subplot(1, 3, 1)
    plt.plot(hist_src)

    plt.subplot(1, 3, 2)
    plt.plot(hist_norm)

    plt.subplot(1, 3, 3)
    plt.plot(hist_eq)

    cv2.imshow("img", img)
    cv2.imshow("img_norm", img_norm)
    cv2.imshow("img_eq", img_eq)

    # 이진화 하는 방법

    threshold = 100

    thr, bin_img_0 = cv2.threshold(img_eq, threshold, 255, cv2.THRESH_BINARY)
    thr_inv, bin_img_1 = cv2.threshold(img_eq, threshold, 255, cv2.THRESH_BINARY_INV)
    thr_trc, bin_img_2 = cv2.threshold(img_eq, threshold, 255, cv2.THRESH_TRUNC)
    thr_tzr, bin_img_3 = cv2.threshold(img_eq, threshold, 255, cv2.THRESH_TOZERO)
    thr_tzr_inv, bin_img_4 = cv2.threshold(img_eq, threshold, 255, cv2.THRESH_TOZERO_INV)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()








