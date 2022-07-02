import csv
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf, precision=4, suppress=True)


def getFPS(content):
    cap = cv2.VideoCapture(content)
    return cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)


def procVideo(objList, viewPoint, content, outputPath):
    cap = cv2.VideoCapture(content)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print('Total frame: ', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    userLookingObjList = []
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        if frame_idx % 500 == 0:
            print(int(frame_idx / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100), '%', end=" ")
        ret, frame = cap.read()
        if ret:
            # cv2.namedWindow('video player', cv2.WINDOW_NORMAL)
            row = np.where(viewPoint[:, 0] == frame_idx)
            if np.any(viewPoint[row]):
                point = viewPoint[row][0]
                x = point[1]
                y = point[2]
                for ind in objList.index:
                    if objList['Obj_xmin'][ind] < int(w * x) < objList['Obj_xmax'][ind] and objList['Obj_ymin'][
                        ind] < int(h * y) < \
                            objList['Obj_ymax'][ind] and objList['Obj_xmin'][ind] == frame_idx:
                        temp = [frame_idx, objList['Obj_xmin'][ind], objList['Obj_ymin'][ind], objList['Obj_xmax'][ind],
                                objList['Obj_ymax'][ind],
                                objList['Obj_name'][ind], objList['Obj_confidence'][ind], int(w * x), int(h * y)]
                        userLookingObjList.append(temp)
            # cv2.imshow('video player', frame)
            # writer.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # cap.release()
    # cv2.destroyAllWindows()
    userLookingObjPan = pd.DataFrame(userLookingObjList,
                                     columns=['Frame_idx', 'Obj_xmin', 'Obj_ymin', 'Obj_xmax', 'Obj_ymax',
                                              'Obj_name', 'Obj_confidence', 'vp_x', 'vp_y'])
    userLookingObjPan.to_csv(outputPath, index=False)
    print("\n============>>>>>>>>output path: ", outputPath)
    # writer.release()


def LocationCalculate(x, y, z, w):
    X = 2 * x * z + 2 * y * w
    Y = 2 * y * z - 2 * x * w
    Z = 1 - 2 * x * x - 2 * y * y

    a = np.arccos(np.sqrt(X ** 2 + Z ** 2) / np.sqrt(X ** 2 + Y ** 2 + Z ** 2))
    if Y > 0:
        ver = a / np.pi * 180
    else:
        ver = -a / np.pi * 180

    b = np.arccos(X / np.sqrt(X ** 2 + Z ** 2))
    if Z < 0:
        hor = b / np.pi * 180
    else:
        hor = (2. - b / np.pi) * 180

    return hor / 360, (90 - ver) / 180


def getViewPoint(exp, user, video):
    path = 'Formated_Data\\Experiment_' + str(exp)
    content = path + '\\Contents\\' + str(video) + '.mp4'
    outputPath = path + '\\' + str(user) + '\\ViewPointWithObj_Video' + str(video) + '.csv'
    FPS, FRAMES = getFPS(content)
    print("============>>>>>>>>processing video: ", content)
    tempPath = path + '\\' + str(user) + '\\video_' + str(video) + '.csv'
    print(tempPath)
    points = np.empty((0, 3))
    with open(tempPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            viewPoint = LocationCalculate(float(row[2]), float(row[3]), float(row[4]), float(row[5]))
            frame = int(float(row[1]) * FPS)
            point = np.array([frame, viewPoint[0], viewPoint[1]])
            if frame != 0 and points.shape[0] == 0:
                foo = 1
            elif points.shape[0] == 0 or points[points.shape[0] - 1][0] < point[0]:
                points = np.vstack([points, point])
    procVideo(points, content, outputPath)


def main():
    for exp in range(1, 3):
        for video in range(0, 8):
            objList = pd.read_csv('Formated_Data\\Experiment_1\\Contents\\obj_video' + str(video) + '.csv')
            for user in range(1, 49):
                getViewPoint(objList, exp, user, video)
