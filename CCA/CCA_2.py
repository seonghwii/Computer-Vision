import cv2
import numpy as np

# binary img example
bin_img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
                    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

rows = bin_img.shape[0]
cols = bin_img.shape[1]

def dfs(bin_img, group_img, cur_x, cur_y, lable): #깊이

    #스택에 현재 좌표를 넣는다.
    stack = [[cur_x, cur_y]]

    dx = [-1, 0, 1, -1, 1, -1, 0, 1]  #순서대로 볼 때 x값의 차를 적어준다. (자기자신은 쓰지 않는다.)
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]  #위와 동일

    # stack이 빌 때까지 돈다.
    while stack:
        x, y = stack.pop()
        # 하나를 꺼내서 방문한 것 ==> label로 표시한다.
        group_img[y, x] = lable

        #인접 픽셀에 방문 가능한 곳이 있다면(값이 있으면) 스택에 넣는다.
        # x-1, y-1 | x, y-1 | x+1, y-1
        # x-1, y   | x, y   | x+1, y
        # x-1, y+1 | x, y+1 | x+1, y+1

        for i in range(8):
            next_x = x + dx[i] #현재 x좌표에 x값의 차들을 넣어준다.(순서대로 돌아간다.)
            next_y = y + dy[i]

            if next_x < 0 or next_x >= cols or next_y < 0 or next_y >= rows:
                continue


            #이미지에 값이 있고 방문 전(group_img ==> lable)이면 stack에 push
            if bin_img[next_y, next_x] != 0 and group_img[next_y, next_x] == 0:
                stack.append([next_x, next_y])

if __name__ == '__main__':
    # 손글씨 이미지 이진화 및 시각화
    src = cv2.imread("hand_write.jpg")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    th, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    print(th)
    bin_img = 255 - bin_img

    rows, cols = gray.shape
    group_img = bin_img.copy()


    """
    label = 0
    
    ### 1) flood fill 사용 ###
    for y in range(rows):
        for x in range(cols):
            if group_img[y, x] ==255:
                label += 1
                cv2.floodFill(group_img, None, (x, y), label)
    
    
    ### 2) ConnectedComponents 사용 ###
    
    #16 바이트로 바꿔준다.
    n_group, label_img = cv2.connectedComponents(bin_img, labels=None, connectivity=8, ltype=cv2.CV_16U)

    ## 그룹에 따라 다른 컬러로 보여줌
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
    group_img = cv2.cvtColor(group_img, cv2.COLOR_GRAY2BGR)

    for y in range(rows):
        for x in range(cols):
            # 3채널 값 동일하므로 하나만 비교하면 됨
            if group_img[y, x] != 0:
                color_idx = group_img[y, x] % len(colors)
                group_img[y, x] = colors[color_idx]
    """

    ### connectedComponentsWithStats 사용 ###
    # 블롭 구하기
    n_blob, label_img, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

    # 블롭을 화면에 보여줌
    show_img = src.copy()

    # n_blob : 0은 배경이므로 뺀다.
    for i in range(1, n_blob):
        x, y, w, h, area = stats[i]  # stats는 n_blob 갯수만큼 나온다.

        # 너무 작은 블롭은 제외한다.
        if area > 50:
            cv2.rectangle(show_img, (x, y, w, h), (255, 0, 255), thickness=2)



