import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

score_mat = pd.read_csv("score_matrix.csv")
ax = sn.heatmap(score_mat, annot=True, vmin=0, vmax=1000, fmt=".1f")
ax.tick_params(length=0)
b, t = plt.ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
plt.ylim(b, t)  # update the ylim(bottom, top) values
plt.title("AIC Score Matrix")
ax.get_figure().savefig("score_matrix.eps", format="eps", dpi=600, bbox_inches="tight")
plt.show()
