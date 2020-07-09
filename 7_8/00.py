# %% [markdown]
import matplotlib.pyplot as plt
import numpy as np
sampleData1 = np.array([[166, 58.7], [176.0, 75.7],
                        [171.0, 62.1], [173.0, 70.4], [169.0, 60.1]])
print(sampleData1)
# %% [markdown]
for p in sampleData1:
    plt.scatter(p[0], p[1], c='k', s=50)
plt.grid()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
# %% [markdown]
for p in sampleData1:
    plt.scatter(p[0], p[1], c='k', s=50)
plt.grid()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
