# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os

imdb_dir = "./aclImdb"
train_dir = os.path.join(imdb_dir, "train")
labels = []
texts = []
for label_type in ["neg", "pos"]:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == ".txt":
            f = open(os.path.join(dir_name, fname), encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == "neg":
                labels.append(0)
            else:
                labels.append(1)


# %%
# 많은 훈련데이터
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100
training_samples = 20000
validation_samples = 2000
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("%s개의 고유한 토큰을 찾았습니다." % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print("데이터 텐서의 크기", data.shape)
print("레이블 텐서의 크기", labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples : training_samples + validation_samples]
y_val = labels[training_samples : training_samples + validation_samples]
x_test = data[training_samples + validation_samples :]
y_test = labels[training_samples + validation_samples :]


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding

embedding_dim = 100
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()


# %%
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(
    x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val)
)
model.save_weights("pre_trained_glove_model.h5")


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
plt.xlabel("Epochs")
plt.ylabel("acc")
plt.legend()

plt.figure()  # 2개의 그래프로 분리

plt.plot(epochs, loss, "rx", label="Training loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# %%
model.load_weights("pre_trained_glove_model.h5")
model.evaluate(x_test, y_test)


# %%
model.predict(x_test[:10])


# %%
print(y_test[:10])
