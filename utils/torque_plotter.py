from matplotlib import pyplot as plt
import numpy as np

torques = np.load('save_torques_09:49:09.npy')

plt.plot(np.linspace(0, torques.shape[0]), torques, linestyle="-", label="Measured")
plt.set_title("Joint Torques")
plt.set_xlabel("Steps")
plt.set_ylabel("Torque (Nm)")

plt.tight_layout()
plt.show()