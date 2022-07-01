import csv
import cv2
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf, precision=4, suppress=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
exp = 2
user = 15
video = 0


def getFPS(content):
    cap = cv2.VideoCapture(content)
    return cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)


def procVideo(viewPoint, content, outputPath):
    print(outputPath)
    cap = cv2.VideoCapture(content)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print('Total frame: ', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    # writer = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (int(w), int(h)))
    userLookingObjList = []
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        if frame_idx % 200 == 0:
            print(int(frame_idx / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100), '%')
        ret, frame = cap.read()
        if ret:
            # cv2.namedWindow('video player', cv2.WINDOW_NORMAL)
            results = model(frame)
            row = np.where(viewPoint[:, 0] == frame_idx)
            if np.any(viewPoint[row]):
                point = viewPoint[row][0]
                x = point[1]
                y = point[2]
                center_coordinates = (int(w * x), int(h * y))
                frame = cv2.circle(frame, center_coordinates, 5, (0, 255, 0), 10)
                data = results.pandas().xyxy[0]  # im predictions (pandas)
                for ind in data.index:
                    if data['xmin'][ind] < int(w * x) < data['xmax'][ind] and data['ymin'][ind] < int(h * y) < \
                            data['ymax'][ind]:
                        # print(data['xmin'][ind], data['ymin'][ind], data['xmax'][ind], data['ymax'][ind],
                        #      data['name'][ind], data['confidence'][ind])
                        # frame = cv2.rectangle(frame, (int(data['xmin'][ind]), int(data['ymin'][ind])),
                        #                       (int(data['xmax'][ind]), int(data['ymax'][ind])),
                        #                       (255, 0, 0), 5)
                        # frame = cv2.putText(frame, data['name'][ind],
                        #                     (int(data['xmin'][ind]) - 10, int(data['ymin'][ind]) - 10),
                        #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
                        temp = [frame_idx, data['xmin'][ind], data['ymin'][ind], data['xmax'][ind], data['ymax'][ind],
                                data['name'][ind], data['confidence'][ind], int(w * x), int(h * y)]
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
    print("============>>>>>>>>output path: ", outputPath)
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


def getViewPoint():
    path = 'Formated_Data\Experiment_' + str(exp)
    content = path + '\\Contents\\' + str(video) + '.mp4'
    outputPath = path + '\\output\\User_' + str(user) + '_ViewPointWithObj_Video' + str(video) + '.csv'
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
    getViewPoint()


if __name__ == '__main__':
    main()
