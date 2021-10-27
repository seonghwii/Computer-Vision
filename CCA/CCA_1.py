"""
[Flood-Fill 알고리즘]
-> 다차원 배열의 어떤 칸과 연결된 영역을 찾는 알고리즘
-> 그림 도구의 채우기 기능 등을 구현할 때 사용한다.

[BFS]
-> 너비 우선 탐색(가까운 노드들부터 우선 탐색)
[알고리즘]
==> 시작 노드를 큐에 삽입 후 방문 처리
==> 큐에서 노드를 pop
==> 꺼낸 노드를 인접한 큐에 노드를 삽입
==> 더 이상 수행할 수 없을 때까지 반복

[DFS]
-> 깊이 우선 탐색(가장 깊은 부분 우선 탐색)
[알고리즘]
==> 시작 노드를 stack에 삽입 후 방문 처리
==> stack에서 노드를 pop
==> pop한 노드의 인접 노드를 스택에 삽입
==> 더 이상 수행할 수 없을 때까지 반복
"""
import numpy as np
import cv2

# 1. Flood-Fill 구현
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

group_img = np.zeros((rows, cols))


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

    # 손글씨 이미지 이진화 및 시각화 [실습]
    src = cv2.imread("hand_write.jpg")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    th, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    bin_img = 255 - bin_img

    cv2.imshow("bin", bin_img)
    cv2.imshow("gray", gray)

    rows = src.shape[0]
    cols = src.shape[1]

    group_img = np.zeros((rows, cols), dtype=np.uint8)
    label = 0

    for y in range(rows):
        for x in range(cols):
            if bin_img[y, x] != 0 and group_img[y, x] == 0:
                label += 1
                dfs(bin_img, group_img, x, y, label)

    ## 그룹에 따라 다른 컬러 씌워줌
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
    group_img = cv2.cvtColor(group_img, cv2.COLOR_GRAY2BGR)
    
    # 이미지 전체를 도는 for문
    for y in range(rows):
        for x in range(cols):
            # 3채널 값 동일하므로 하나만 비교하면 됨
            if group_img[y, x, 0] != 0:
                color_idx = group_img[y, x, 0] % len(colors)
                group_img[y, x] = colors[color_idx]

    cv2.imshow("group", group_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

