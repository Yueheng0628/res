import torch
import cv2
import pandas as pd

model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

for i in range(2, 9):
    ObjList = []
    content = 'Formated_Data\\Experiment_1\\Contents\\' + str(i) + '.mp4'
    outputPath = 'Formated_Data\\Experiment_1\\Contents\\obj_video' + str(i) + '.csv'
    cap = cv2.VideoCapture(content)
    print("============>>>>>>>>path: ", content)
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        if frame_idx % 500 == 0:
            print(int(frame_idx / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100), '%', end=" ")
        ret, frame = cap.read()
        # cv2.namedWindow('video player', cv2.WINDOW_NORMAL)
        results = model(frame)
        # results.print()
        # results.xyxy[0]  # im predictions (tensor)
        data = results.pandas().xyxy[0]  # im predictions (pandas)
        for ind in data.index:
            temp = [frame_idx, data['xmin'][ind], data['ymin'][ind], data['xmax'][ind], data['ymax'][ind],
                    data['name'][ind], data['confidence'][ind]]
            ObjList.append(temp)
        # cv2.imshow('video player', frame)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break
    # cap.release()
    # cv2.destroyAllWindows()
    ObjPan = pd.DataFrame(ObjList,
                          columns=['Frame_idx', 'Obj_xmin', 'Obj_ymin', 'Obj_xmax', 'Obj_ymax',
                                   'Obj_name', 'Obj_confidence'])
    ObjPan.to_csv(outputPath, index=False)
    print("\n============>>>>>>>>output path: ", outputPath)
