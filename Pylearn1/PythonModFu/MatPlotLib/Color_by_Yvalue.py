import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 2.0, 0.01)
s = np.sin(2 * np.pi * x)

upper = 0.5
lower = -0.5

supper = np.ma.masked_where(s <= upper-1, s)
slower = np.ma.masked_where(s >= lower+1, s)
smiddle = np.ma.masked_where((s <= lower) | (s >= upper), s)

plt.plot(x, smiddle, 'r*', x, slower, 'g^', x, supper, 'b-',)
plt.show()
