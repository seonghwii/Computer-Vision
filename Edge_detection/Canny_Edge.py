"""
[Canny Edge]
: maxval중에서 minval보다는 크고 연결되어 있는 것 ==> Edge 취급

- Canny Edge의 특징
1. 가우시안 필터 사용
2. Gradient 계산(소벨 사용)
"""


import cv2

if __name__ == '__main__':
    src = cv2.imread("Rena.png", cv2.IMREAD_GRAYSCALE)




    #(src, minval, maxval)
    #===> minval가 너무 낮으면 너무 많은 edge가 드러난다.
    #===> maxval가 너무 크면 edge가 많이 나타나지 않는다.

    edge = cv2.Canny(src, 50, 200)
    edge1 = cv2.Canny(src, 100, 200)
    edge2 = cv2.Canny(src, 120, 300)
    edge3 = cv2.Canny(src, 100, 500)


    cv2.imshow("edge", edge)
    cv2.imshow("edge1", edge1)
    cv2.imshow("edge2", edge2)
    cv2.imshow("edge3", edge3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()