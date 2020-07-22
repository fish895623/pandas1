# NOTE 미완성작

# %%
# 1. 데이터 준비하기
from IPython import get_ipython

# 코드 사전 정의

code2idx = {
    'c4': 0,
    'd4': 1,
    'e4': 2,
    'f4': 3,
    'g4': 4,
    'a4': 5,
    'b4': 6,
    'c2': 7,
    'd2': 8,
    'e2': 9,
    'f2': 10,
    'g2': 11,
    'a2': 12,
    'b2': 13
}

# %%
idx2code = {v: k for k, v in code2idx.items()}
print(idx2code)

# %%
seq = [
    'c4', 'c4', 'g4', 'g4', 'a4', 'a4', 'g2', 'f4', 'f4', 'e4', 'e4', 'd4', 'd4', 'c2',
    'g4', 'g4', 'f4', 'f4', 'e4', 'e4', 'd2', 'g4', 'g4', 'f4', 'f4', 'e4', 'e4', 'd2',
    'c4', 'c4', 'g4', 'g4', 'a4', 'a4', 'g2', 'f4', 'f4', 'e4', 'e4', 'd4', 'd4', 'c2'
]
# 데이터셋 생성 함수
import numpy as np


def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq) - window_size):
        subset = seq[i:(i + window_size + 1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)


# 2. 데이터셋 생성하기
dataset = seq2dataset(seq, window_size=6)
print(dataset)

# %%
# 0. 사용할 패키지 불러오기
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.utils import np_utils

# 랜덤시드 고정시키기
np.random.seed(5)


# 손실 이력 클래스 정의
class LossHistory(tensorflow.keras.callbacks.Callback):
    def init(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# 입력(X)과 출력(Y) 변수로 분리하기
x_train = dataset[:, 0:4]
y_train = dataset[:, 4]

max_idx_value = 13

# 입력값 정규화 시키기
x_train = x_train / float(max_idx_value)

x_train = np.reshape(x_train, (36, 4, 1))

# 라벨값에 대한 one-hot 인코딩 수행
y_train = np_utils.to_categorical(y_train)

one_hot_vec_size = y_train.shape[1]

print("one hot encoding vector size is ", one_hot_vec_size)

# 3. 모델 구성하기
model = Sequential()
model.add(LSTM(128, input_shape=(6, 1)))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = LossHistory()  # 손실 이력 객체 생성
history.init()

# 5. 모델 학습시키기
model.fit(x_train,
          y_train,
          epochs=2000,
          batch_size=10,
          verbose=2,
          callbacks=[history])

# %%
# 6. 학습과정 살펴보기
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 7. 모델 평가하기
scores = model.evaluate(x_train, y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# %%
# 8. 모델 사용하기

pred_count = 50  # 최대 예측 개수 정의

# 한 스텝 예측

seq_out = ['g8', 'e8', 'e4', 'f8']
pred_out = model.predict(x_train)

for i in range(pred_count):
    idx = np.argmax(pred_out[i])  # one-hot 인코딩을 인덱스 값으로 변환
    seq_out.append(idx2code[idx])  # seq_out는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장

print("one step prediction : ", seq_out)

# %%
# 곡 전체 예측

seq_in = ['g8', 'e8', 'e4', 'f8']
seq_out = seq_in

seq_in = [code2idx[it] / float(max_idx_value)
          for it in seq_in]  # 코드를 인덱스값으로 변환 동시에 정규화!

for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1, 4, 1))  # batch_size, feature
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)  # 새로운 음표가 들어왔으니 이전 첫 음표는 삭제 하여 다음 입력으로 사용!

print("full song prediction : ", seq_out)

# %%
max_idx_value

# %%
