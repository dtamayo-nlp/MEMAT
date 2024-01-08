import random
import numpy as np

random.seed(1234)
combination = [(random.randint(0,31),random.randint(0,70)) for i in range(16)]
print(combination)
np.save("./random_ind.npy", combination)