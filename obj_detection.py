import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

for i in range(0, 1):
    content = 'Formated_Data\Experiment_2/Contents/' + str(i) + '.mp4'
    cap = cv2.VideoCapture(content)
    print("============>>>>>>>>path: ", content)
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        cv2.namedWindow('video player', cv2.WINDOW_NORMAL)
        results = model(frame)
        # results.print()
        # results.xyxy[0]  # im predictions (tensor)
        data = results.pandas().xyxy[0]  # im predictions (pandas)
        for ind in data.index:
            if data['confidence'][ind] > 0.5:
                print(data['xmin'][ind], data['ymin'][ind], data['xmax'][ind], data['ymax'][ind], data['name'][ind], data['confidence'][ind])
                frame = cv2.rectangle(frame, (int(data['xmin'][ind]), int(data['ymin'][ind])),
                                      (int(data['xmax'][ind]), int(data['ymax'][ind])),
                                      (255, 0, 0), 5)
                frame = cv2.putText(frame, data['name'][ind], (int(data['xmin'][ind])-10, int(data['ymin'][ind])-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('video player', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
