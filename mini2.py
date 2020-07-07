# %% import
import pandas as pd
import numpy as np
from numpy import NaN
from sklearn.metrics import accuracy_score
import math
import random

# %%
# 1 pandas형식으로 전환
header_list = ["q", "w", "e", "r"]
data = pd.read_csv("iris.csv", names=header_list)


# %%
# 2 Binning 수행하여 컬럼 추가하기
# %%
# 3 get_dummies()함수를 활용
dummy_data = pd.get_dummies(data)

# %%
# 4 NaN 데이터 확인, 있다면 삭제, 평균을 구하여 대체하여 업데이트
# NOTE NaN 삭제
# data1 = data.dropna()
# data.replace(NaN, data["color"].mean())

# %%
# 5 Nomalization을 적용
# %%
# 6 결과 3가지를 모델에 적용하여 정확도를 구하기

# %% [markdown]
int(random.uniform(1, 12))
