import numpy as np
import matplotlib.pyplot as plt
data_x = np.linspace(0,10,30)
data_y = data_x*3 + 7 + np.random.normal(0,1,30)

plt.scatter(data_x,data_y)
plt.show()