# %% [markdown]
from keras_preprocessing.text import Tokenizer


samples = [
    "the cat sat on the mat",
    "the dog ate my homework",
    "the the ate ate dog dog",
    "가 나 다",
]

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode="binary")
word_index = tokenizer.word_index
print("%s 개의 토큰" % len(word_index))
word_index

# %%
sequences

# %%
one_hot_results.shape
# %%
one_hot_results[:10, :10]
# %%
samples = ["그 고양이는 맽 위에 앉았다", "그 개는 숙제를 먹었다"]
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode="binary")
word_index = tokenizer.word_index
print("%s 개의 토큰" % len(word_index))


# %%
sequences

# %%
one_hot_results.shape
# %%
one_hot_results[:10, :10]