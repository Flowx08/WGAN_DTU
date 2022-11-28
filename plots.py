import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

fid_gansn = genfromtxt('results/celeba_gansn.csv', delimiter=',')[1:]
fid_wgangp = genfromtxt('results/celeba_wgangp.csv', delimiter=',')[1:]
print(fid_gansn)
print(fid_wgangp)

x = np.arange(0, len(fid_gansn), step=1) + 1
plt.plot(x, fid_gansn, label="GAN-SN")
plt.plot(x, fid_wgangp, label="WGAN-GP")
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID plot during training of the models")
plt.legend()
#plt.show()
plt.savefig("results/fid_training_comparasion.png")



