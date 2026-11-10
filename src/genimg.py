import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

loss_history = pd.read_csv("../results/csv/loss_history.csv")
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(loss_history["iteration"], loss_history["loss"], color="blue", linewidth=2)
ax.set_title("Loss History")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.grid(True, linestyle="--", alpha=0.5)
fig.savefig("../results/plots/loss_history.png", dpi=300, bbox_inches="tight")