import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

conf_mat = np.load("conf_mat.npy")
df_cm = pd.DataFrame(
    conf_mat,
    index=["RW", "CK", "RWCK", "NWSLS"],
    columns=["RW True", "CK True", "RWCK True", "NWSLS True"],
)
ax = sn.heatmap(df_cm, annot=True, vmin=0, vmax=1)
ax.tick_params(length=0)
b, t = plt.ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
plt.ylim(b, t)  # update the ylim(bottom, top) values
plt.title("Model Recovery Confusion Matrix")
ax.get_figure().savefig("recovery.eps", format="eps", dpi=600, bbox_inches="tight")
plt.show()
