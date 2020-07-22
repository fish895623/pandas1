# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32
print("데이터 로딩...")
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), "train 시퀀스")
print(len(input_test), "test 시퀀스")
print("시퀀스 패딩 (samples x time)")
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print("input_train 크기", input_train.shape)
print("input_test 크기", input_test.shape)


# %%
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation="sigmoid"))
model.summary()


# %%
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    input_train, y_train, epochs=10, batch_size=128, validation_split=0.2
)


# %%
import matplotlib.pyplot as plt

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="validation acc")
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()  # 2개의 그래프로 분리

plt.plot(epochs, loss, "rx", label="Training loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Loss/acc")
plt.legend()
plt.show()


# %%
test_mae_score, test_mae_score = model.evaluate(input_test, y_test)
print(test_mae_score)


# %%

