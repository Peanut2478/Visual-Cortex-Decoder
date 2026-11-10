import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path

cm = np.array([
    [11, 1, 2, 0, 3, 0, 0, 0],
    [1, 12, 1, 0, 0, 0, 0, 0],
    [1, 0, 8, 0, 3, 0, 4, 1],
    [0, 0, 0, 10, 0, 0, 0, 3],
    [0, 0, 3, 0, 9, 2, 0, 0],
    [3, 1, 0, 1, 1, 11, 0, 0],
    [1, 0, 6, 1, 0, 1, 4, 0],
    [0, 0, 0, 4, 0, 0, 0, 11]
])

directions = np.arange(0,360,45)
opp_err = np.array([cm[i,(i+4)%8] for i in range(8)])
err_rates = opp_err / cm.sum(axis=1) * 100
plt.figure(figsize=(9,5))
bars = plt.bar(
    directions.astype(str),
    err_rates,
    color="steelblue"
)
for bar, count in zip(bars, opp_err):
    plt.text(
        bar.get_x() +bar.get_width() / 2,
        bar.get_height()+1,
        str(count),
        ha="center"
    )
plt.xlabel("Act Stimulus Direction (Degrees)")
plt.ylabel("Opp-dir Err Rate (%)")
plt.title("Neural Network: 180-Degree Decoding Errors")
plt.ylim(0,max(err_rates)+12)
plt.tight_layout()

Path("results/figures").mkdir(parents=True,exist_ok=True)
plt.savefig("results/figures/opposite_direction_errors.png",dpi=300)
plt.show()
print("Figure saved")