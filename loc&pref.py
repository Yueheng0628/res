import collections
import csv
import cv2
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt


def getPref(inF, output):
    df = pd.read_csv(input)
    temp = df['Obj_name'].value_counts().to_frame()
    obj = temp.index.tolist()
    freq = temp['Obj_name'].tolist()
    plt.figure(figsize=(9, 6))
    plt.barh(obj, freq)
    for index, value in enumerate(freq):
        plt.text(value, index, str(value))
    plt.savefig(output)


def getloc(inF, output):
    df = pd.read_csv(inF)
    alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    res = []
    for ind in df.index:
        xmin = df['Obj_xmin'][ind]
        xmax = df['Obj_xmax'][ind]
        ymin = df['Obj_ymin'][ind]
        ymax = df['Obj_ymax'][ind]
        x = df['vp_x'][ind]
        y = df['vp_y'][ind]
        unitX = (xmax - xmin) / 3
        unitY = (ymax - ymin) / 3
        maxPointList = []
        for i in range(1, 4):
            for j in range(1, 4):
                tempxmin = xmin + (j - 1) * unitX
                tempymin = ymin + (i - 1) * unitY
                tempxmax = xmin + j * unitX
                tempymax = ymin + i * unitY
                maxPointList.append((tempxmin, tempymin, tempxmax, tempymax))
        for index in range(0, 9):
            if maxPointList[index][0] < x <= maxPointList[index][2] \
                    and maxPointList[index][1] < y <= maxPointList[index][3]:
                res.append(alph[index])
    frequency = collections.Counter(res)
    print(frequency)


def main():
    inF = 'Formated_Data\Experiment_2\\output\\User_1_ViewPointWithObj_Video0.csv'

    # getPref(inF, output='Formated_Data\Experiment_1\\output\\User10_Video5_ViewObjfreq.png')

    getloc(inF, output='Formated_Data\Experiment_1\\output\\User10_Video5_ViewObjloc.png')


if __name__ == '__main__':
    main()
