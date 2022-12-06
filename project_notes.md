- Explaining GAN and Explain the instability of training
	- Mode collapse

- Model Improvements
	- 1. WGAN weight clipping (from the original paper)
	- 2. WGAN with Gradient penalty
	- 3. Spectral Normalization

- Architecture:
	- Inspired by DCGAN paper
		- Used convolutional layers with increasing filters
		- Uses batch normalization in generator and layer normalization in discriminator
		- Tanh activation for the generator
		- Sigmoid or linear activation for dicriminator based on if its vanilla GAN or WGAN
	
- Explaining the experiments
	- Pre-processing:
		- We scaled all the images to 32x32 pixels for MNIST, FashionMNIST and CelebA. In this way we could you a single architecture for all the datasets and less resources are needed.
		- We normalized all the images.
	- MNIST, Fashion MNIST, CelebA
		- Increse level of difficulty, CelebA has more dimensions (3x32x32) and MNIST is (1x28x28).
		- We tried CIFAR10, but the model did not generate good images after 100 epochs so we tested our model on CelebA to see if it was because of the difficulty of CIFAR10 and we got good results on CelebA.
		- CelebA -> 10 epoch of training.
		- MNIST, FashionMNIST -> 15 epoch of training
	- Models: GAN, GAN-SN, WGAN-GP
		- We didn' use WGAN-SN because first experimental results showed that it wasn't performing well, but applying SN to GAN improved the performance considerably.

- Explaining the results
	- Explaining FID
	- We saw that the FID is descreasing over the epochs of training as expected
	- We saw that Vanilla GAN overall performes worse than WGAN-GP and GAN-SN.
	- Even tho WGAN-GP has the best performances, it takes about 3 times more time to train than the other models, because in the WGAN we train the discriminator 5 times more frequently than the generator.


