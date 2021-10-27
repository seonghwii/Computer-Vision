"""
[실습 예제]
백준 2667번 문제
"""

import numpy as np



def dfs(cur_x, cur_y, lable, src, ans, value):
    stack = [[cur_x, cur_y]]

    rows, cols = src.shape[0], src.shape[1]

    dx = [0, 0, -1, 1]
    dy = [-1, 1, 0, 0]

    cnt = 0

    while stack:
        x, y = stack.pop()

        if ans[y, x] == 0:
            ans[y, x] = lable
            cnt += 1

        for i in range(4):
            next_x = x + dx[i]
            next_y = y + dy[i]


            #예외처리
            if next_x < 0 or next_x >= cols or next_y < 0 or next_y >= rows:
                continue
            #방문을 아직 안했으면 stack에 append해준다.
            if src[next_y, next_x] == value and ans[next_y, next_x] == 0:
                stack.append([next_x, next_y])


    return cnt


bin_img1 = np.array([[0, 1, 1, 0, 1, 0, 0],
                     [0, 1, 1, 0, 1, 0, 1],
                     [1, 1, 1, 0, 1, 0, 1],
                     [0, 0, 0, 0, 1, 1, 1],
                     [0, 1, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 0],
                     [0, 1, 1, 1, 0, 0, 0]])
def solution1():
    ans1 = np.zeros(bin_img1.shape, dtype=np.uint8)


    lable1 = 0
    for y in range(bin_img1.shape[0]):
        for x in range(bin_img1.shape[1]):
            if bin_img1[y, x] == 1 and ans1[y, x] == 0:
                lable1 += 1
                dfs(x, y, lable1, bin_img1, ans1, 1)

    print(f'단지의 수는 {lable1}개입니다.')

    b = np.count_nonzero(ans1 == 1)
    c = np.count_nonzero(ans1 == 2)
    d = np.count_nonzero(ans1 == 3)
    # count_nonzero == : numpy에서 zero가 아닌 것(ans1에서 값이 1인 갯수를 세주는 함수)

    print(b)
    print(c)
    print(d)
