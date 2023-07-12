# AI-models
Some AI models which I have implemented to enforce my coding skills
## 1. Denoising Diffusion Probabilistic Models (DDPMs)
Diffusion probabilistic models:
"A diffusion probabilistic model is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time" or Diffusion probabilistic models are parameterized Markov chains models trained to gradually denoise data.
Simply put, diffusion models can generate data similar to the ones they are trained on. If the model trains on images of cats, it can generate similar realistic images of cats. In recent years DDPMs are outperforming the GANs in many de-noising applications


## GANs ( Generative adversarial network ): 
In a GAN, two neural networks contest with each other in the form of a zero-sum game, where one agent's gain is another agent's loss. GANs has two neural networks. One is Discriminator and the other one is Generator. Generator generates fake images and Discriminator discriminates between the fake and given real image. If the generator can generate an image which the discriminator is not able to separate between the real and fake, we call the neural network fully trained. Generators receive a random (gaussian distribution ) image and generate the fake image. 
