import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

exp = 2
user = 3
video = 3
path = 'Formated_Data\Experiment_' + str(exp)
content = path + '\\' + str(user) + '\\video_' + str(video) + '.csv'

df = pd.read_csv(content)
# count = 1
# for ind in df.index:
#     if 5*(count-1)< df['PlaybackTime'][ind] < 5*count:
time = df['PlaybackTime'].to_numpy()[50:450]
uqx = df['UnitQuaternion.x'].to_numpy()[50:450]
uqy = df['UnitQuaternion.y'].to_numpy()
uqz = df['UnitQuaternion.z'].to_numpy()
uqw = df['UnitQuaternion.w'].to_numpy()

# plt.figure(figsize=(9, 6))
# plt.scatter(time, uqx)
# plt.xlabel('time')
# plt.ylabel('uq_x')
# plt.show()

sp = np.fft.fft(uqx)
freq = np.fft.fftfreq(uqx.shape[-1])
plt.plot(freq, sp.real)
plt.show()
