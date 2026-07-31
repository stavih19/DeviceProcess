import numpy as np
import matplotlib.pyplot as plt

data = np.load("Teramount_Home_Assignment/15_files_task_1/0.npy", allow_pickle=False)

plt.figure(figsize=(10, 8))
plt.imshow(data, cmap="viridis", aspect="auto")
plt.colorbar(label="Height")
plt.title(f"Shape: {data.shape}")
plt.show()