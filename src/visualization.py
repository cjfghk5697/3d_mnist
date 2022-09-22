# -*- coding: utf-8 -*-
"""입체_데이터 시각화.ipynb

## 패키지 준비
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')

# %cd "/content/drive/MyDrive/data/3d data"

!pip install plotly

import plotly.express as px

import h5py

"""## 데이터 불러오기"""

all_points = h5py.File('./train.h5', 'r')
test_all = h5py.File('./test.h5', 'r')

"""## 이하 plotly 패키지 활용"""

key = '2'

x = all_points[key][:, 0]
y = all_points[key][:, 1]
z = all_points[key][:, 2]

px.scatter_3d(x=x, y=y, z=z, opacity = 0.8)

key = '50413'

x = test_all[key][:, 0]
y = test_all[key][:, 1]
z = test_all[key][:, 2]

px.scatter_3d(x=x, y=y, z=z, opacity = 0.8)

key = '50600'

x = test_all[key][:, 0]
y = test_all[key][:, 1]
z = test_all[key][:, 2]

px.scatter_3d(x=x, y=y, z=z, opacity = 0.8)

