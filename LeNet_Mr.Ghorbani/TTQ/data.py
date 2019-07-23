import os
import torch
import torchvision
from torchvision import transforms

batch_size = 100
mnist_folder = os.path.join(os.path.dirname(__file__), 'mnist')
train_dataset = torchvision.datasets.MNIST(root=mnist_folder,
                                           train=True,
                                           transform=transforms.Compose([
                                             transforms.Resize((32, 32)),
                                            transforms.ToTensor()]),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=mnist_folder,
                                          train=False,
                                          download=True,
                                          transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=256,
                                           shuffle=True,
                                           num_workers=8)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1024,
                                          num_workers=8)
