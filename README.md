# AI-models
Some AI models which I have implemented to enforce my coding skills
## 1. Denoising Diffusion Probabilistic Models (DDPMs)
Diffusion probabilistic models:
"A diffusion probabilistic model is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time" or Diffusion probabilistic models are parameterized Markov chains models trained to gradually denoise data.
Simply put, diffusion models can generate data similar to the ones they are trained on. If the model trains on images of cats, it can generate similar realistic images of cats. In recent years DDPMs are outperforming the GANs in many de-noising applications
Some Examples obtained by slightly adjusted DDPM model on  torchvision.datasets.FGVCAircraft dataset: 
![image](https://github.com/metenes/AI-models/assets/91368249/17d084db-106a-42ce-bea4-312174c1c03e)
![image](https://github.com/metenes/AI-models/assets/91368249/c7637224-a512-4cfb-835b-50fc5769ca2c)
![image](https://github.com/metenes/AI-models/assets/91368249/60daf6fe-3b72-4e52-a111-7afb397dcfb5)

Nevertheless, this model is not suitable for more complex images than standar MNIST or CIFAR100 datasets,
## GANs ( Generative adversarial network ): 
In a GAN, two neural networks contest with each other in the form of a zero-sum game, where one agent's gain is another agent's loss. GANs has two neural networks. One is Discriminator and the other one is Generator. Generator generates fake images and Discriminator discriminates between the fake and given real image. If the generator can generate an image which the discriminator is not able to separate between the real and fake, we call the neural network fully trained. Generators receive a random (gaussian distribution ) image and generate the fake image. 

## DDIMs ( Denoising Diffusion Implict Models ): 
Implicit probabilistic models, also known as implicit generative models or implicit models, are a class of machine learning models used for generating data. Unlike explicit probabilistic models, which directly model the probability distribution of the data, implicit models do not explicitly represent the probability distribution. Instead, they learn to generate samples from the data distribution without directly modeling it.
The key characteristic of implicit probabilistic models is that they do not explicitly compute the probability of generating a specific data point. Instead, they rely on a learning process that implicitly encodes the distribution of the data in their learned parameters.
