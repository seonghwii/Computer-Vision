"""
[haarcascade를 이용한 face detection]
"""

import cv2 as cv
import os

file_list= []

# 파일 경로의 모든 이미지 이름을 가져오는 것
def get_all_img_time(root_dir):
    global file_list
    elem = os.listdir(root_dir)
    # root_dir 안에 있는 file name list를 반환한다.


    # directory 안에 있는 file == e
    for e in elem:
        full_path = os.path.join(root_dir, e)
        # join : 모든 경로를 보여주는 함수 | os.path.join(파일 경로, 변수)
        # print(full_path)

        if os.path.isdir(full_path): # directory일 경우 재귀(get_all_time 함수에 다시 넣어준다.)
            get_all_img_time(full_path + "/")

        elif os.path.isfile(full_path): # file일 경우 file_list 에 이름을 넣어준다.
            file_name, ext = os.path.splitext(full_path)
            if ext == ".txt" or ext == ".json" or ext == ".xml":
                continue

            else:
                file_list.append(full_path)

# 즉, get_all_img_time 함수는 모든 file name을 리스트에 넣어준다.


def face_det():
    global file_list
    count = 0 # 파일의 수

    # 이미지, 레이블을 저장할 경로
    save_path = "./pos_result"

    model_file = "./haarcascade_frontalface_alt.xml"
    clf = cv.CascadeClassifier(model_file)
    # CascadeClassifier 사용

    # 모든 이미지 파일에 대해서
    for file in file_list:
        cur_img = cv.imread(file)

        # 이미지 파일을 학습 DB에 저장
        # 저장할 이미지와 레이블의 형식을 지정해줌. 
        # 이미지와 레이블은 확장자의 형식을 제외하고는 형식을 같게 해준다.
        img_name = save_path + "face_{0:05d}".format(count) + ".jpg"
        label_name = save_path + "face_{0:05d}".format(count) + ".txt"
        count += 1

        cv.imwrite(img_name, cur_img) # 파일 저장(저장할 이름(filename), 저장할 이미지)
        fp = open(label_name, "w") # label 해준 텍스트 파일로 저장한다.

        gray = cv.cvtColor(cur_img, cv.COLOR_BGR2GRAY)
        face_rects = clf.detectMultiScale(gray)
        #return 값 : [x, y, w, h] (list)형식

        for rect in face_rects:
            x, y, w, h = rect
            """
            YOLO FORMAT
            x_center = (x + (w/2.)) / cur_img.shape[1] # cur_img.shape[1]로 나눠주는 이유는 normalize하기 위해서이다.
            y_center = (y + (h/2.)) / cur_img.shape[0]
            yolo_w = w / cur_img.shape[1]
            yolo_h = h / cur_img.shape[0]
            """

            cv.rectangle(cur_img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=1)

            # YOLO FORMAT 에서는 CLASS를 꼭 앞에 지정해주어야 한다. CLASS : 0
            # fp.write(f"0 {x} {y} {w} {h}\n")

            fp.write('{x, y, w, h}\n')


        fp.close()

        cv.imshow("src", cur_img)
        key = cv.waitKey(0)
        if key == 27:
            break

        cv.destroyAllWindows()

# 사진을 띄웠을 때 나오는 x, y, w, h를 그림에 나타나게 하는 함수
def show_data():
    # labeling된 폴더
    db_path = "D:/faceDB_labled/" # DB == img, labeling한 파일을 데이터 베이스로 취급한다.
    pos_path = "./"
    neg_path = "./"

    count = 0

    files = os.listdir(db_path)

    for file in files:
        full_file = os.path.join(db_path, file) #db_path랑 file을 조인한다.
        name, ext = os.path.splitext(full_file) # name과 ext(확장자)를 분리한다.

        #확장자가 jpg 파일일 경우
        if ext == ".jpg":
            cur_img = cv.imread(full_file)

            gt_file = name + ".txt"
            fp = open(gt_file, "r")
            line = fp.readline()

            while line:
                cls_name, x_center, y_center, w, h = line.split()
                rect_w = int(float(w) * cur_img.shape[1])
                rect_h = int(float(h) * cur_img.shape[0])
                x = int(float(x_center) * cur_img.shape[1])
                y = int(float(y_center) * cur_img.shape[0])
                x = int(x - (rect_w / 2.) +0.5) # 0.5 : 반올림
                y = int(y - (rect_h / 2.) +0.5)


                count += 1

                X = x+100
                Y = y+100

                det_rect = [X, Y, rect_w, rect_h]

                score = IOU([X, Y, rect_w, rect_h], det_rect)
                print("score", score)

                cv.rectangle(cur_img, (X, Y), (X+rect_w, Y+rect_h), color=(0, 0, 255))
                cv.rectangle(cur_img, (X, Y), (X+rect_w, Y+rect_h), color=(0, 255, 0))

                print(x,y, rect_w, rect_h)
                line = fp.readline()



            fp.close()


def IOU(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    int_width = min(x1+w1, x2+w2) - max(x1,x2) # 교집합 넓이
    int_height = min(y1+h1, y2+h2) - max(y1, y2)

    if int_height <= 0 or int_width <= 0:
        return 0 # 겹치지 않는 것

    area1 = w1 * h1
    area2 = w2 * h2

    int_area = int_width * int_height # 교집합 영역
    union_area = area1 + area2 - int_area #합집합 영역

    return int_area / union_area

if __name__ == '__main__':
    # root_dir = "./positive"
    # get_all_img_time(root_dir)
    # face_det()
    show_data()