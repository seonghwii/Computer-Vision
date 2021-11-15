"""
[Edge detection]
: 명암, 컬러 또는 텍스처의 변화량을 측정 -> 변화량이 큰 곳을 에지로 검출 가능
- 변화량? 미분
1차 미분 ==> f'(x) = f(x+1) - f(x)
2차 미분 ==> f''(x) = f(x+1) + f(x-1) + 2f(x)
--2차 미분을 하는 이유?
: 자연 현상에서는 주로 램프 에지가 나타나는데 그 구간을 미분하면 차가 크지 않기 때문이다.

- 구현 : mask를 씌움(filter)

[convolution]
: filter를 elemental wise로 곱하여 더한다. (CNN에서도 같은 원리 사용)
"""

import cv2
import numpy as np

#kernel == mask(filter)
kernel_y = np.array([[1, 0, -1],
                     [2, 0, -2],
                    [1, 0, -1]])

kernel_x = np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]])

gauss_filter = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]])


if __name__ == '__main__':
    src = cv2.imread("tower.JPG")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    rows, cols = gray.shape
    # 객체를 찾는 박스가 3x3 ==> 가장자리는 찾을 수 없기 때문에 -2를 해서 영점을 맞춰준다.(padding 부분 삭제)
    gradient = np.zeros((rows-2, cols-2), dtype=np.uint8)
    g_x = np.zeros((rows-2, cols-2), dtype=np.uint8)
    g_y = np.zeros((rows-2, cols-2), dtype=np.uint8)

    for y in range(rows-2):
        for x in range(cols-2):
            roi = gray[y:y+3, x:x+3] #찾는 범위(영역 : 3x3 | 가장자리를 찾을 수 없기 때문에 padding해줌)

            """
            [gradient 구현]
            sx = np.sum(kernel_x * roi) # convolution 구현
            sy = np.sum(kernel_y * roi)
            # s => 강도 ==> 루트 x방향^2 + y방향^2
            gradient[y, x] = np.sqrt(sx**2 + sy**2)
            g_x[y, x] = np.sqrt(sx**2 + sx**2)
            g_y[y, x] = np.sqrt(sy**2 + sy**2)
            #
            val = np.sqrt(sx**2 + sy**2)
            if val > 150: #한계치
                gradient[y, x] = val

            else:
                gradient[y, x] = 0
            """

            # [Gaussian Filter 적용]
            # (Bluring 효과: 노이즈 감소 | 표준편차가 클수록 더 큰 스무딩 효과)
            # 3x3 가우스 필터
            gx = np.sum(gauss_filter*roi) #convolution 해주는 것이다.
            gradient[y, x] = gx / 16


    cv2.imshow("gradient", gradient)
    # cv2.imshow("g_x", g_x)
    # cv2.imshow("g_y", g_y)
    cv2.imshow("GRAY", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


