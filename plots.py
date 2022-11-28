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


fid_wgangp = genfromtxt('results/fid_wgangp.csv', delimiter=',')
fid_gansn = genfromtxt('results/fid_gansn.csv', delimiter=',')
fid_gan = genfromtxt('results/fid_gan.csv', delimiter=',')
n=3
r = np.arange(n)
width = 0.20
fig = plt.figure(figsize=(6, 4), dpi=100)
plt.bar(r, fid_wgangp, color = 'b', width = width, edgecolor = 'black', label='WGAN-GP')
plt.bar(r + width, fid_gansn, color = 'y', width = width, edgecolor = 'black', label='GAN-SN')
plt.bar(r + width * 2, fid_gan, color = 'r', width = width, edgecolor = 'black', label='Vanilla GAN')
plt.xlabel("Taskset")
plt.ylabel("FID")
plt.title("FID value of our models on MNIST, FashionMNIST\nand CelebA datasets")
plt.xticks(r + width, ['MNIST','FashionMNIST','CelebA'])
plt.legend()
#plt.show()
plt.savefig("./results/fid_best_comparison.png")
#plt.clf() #clear

plt.show()

