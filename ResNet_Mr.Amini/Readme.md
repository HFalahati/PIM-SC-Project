INQ implementation for ResNet structure
===

MNIST
---
This folder contains two subfolders : **Network** and **Quantization**.

In **Network** folder, I implement ResNet32 based on [original paper](https://arxiv.org/abs/1512.03385).
and In **Quantization** folder, I implement INQ.

Training the network
---
In **Network** folder, just run the `Resnet_orig.py`.
You can change the Network parameters(num_epoches , batch_size ,...) or log parameters(Dataset , weights_dir , log_dir ,...) in `config.py` file.  
Also in `config.py` file, you can change the network depth and construct other ResNet models(ResNet14 , ResNet20 , ... , ResNet<3n+2>(n is an even number)).   

In this case, I have used from [MNIST](http://yann.lecun.com/exdb/mnist/) dataset but you can use any other tiny dataset(Cifar10 , Cifar100 , tinyImageNet ,...). If you want to use MNIST dataset, you should [download](http://yann.lecun.com/exdb/mnist/) it 
and put it into **MNIST_dataset** folder. Otherwise you should write a function in `dataSets.py` file that load your intended dataset and also you should
change the `reshape()` function in `Resnet_orig.py` based on input size of images in dataset.   

`saver.py` file in **Saver** folder responsible for save the network weights in **MNIST_weights** folder when the validation accuracy is improved.   

I run the `Resnet_orig.py` and the following results are obtained :   
```
model : ResNet32
n_epoch : 100
batch_size : 128
optimizer : SGD with Momentum
dataset : MNIST
training accuracy : 1.0000
validation accuracy : 0.9780
testing accuracy : 0.9772
```   

Quantization
---
I implement INQ algorithm in `quantization.py`. To use this file you should ,first, train the network and obtain weights. then copy the network weights 
to **MNIST_weights** folder and then run the `quantization.py`.   
You can change the quantization parameters(bit_width and accumulated_portion) in `config.py`.   

I run the `quantization.py` with 5 epoch retraining and for 3 and 5 bit width, and the following results are obtained :    
```
validation accuracy with 32 bit floating point = 97.80 %

                5 epoch retraining , bit_width = 3             accumulated_portion
        
                iteration 1 : validation_acc = 47.08 %                  0.5
                iteration 2 : validation_acc = 75.68 %                  0.75
                iteration 3 : validation_acc = 89.44 %                  0.875
                iteration 4 : validation_acc = 87.94 %                  1.0
                
                
                
                
                5 epoch retraining , bit_width = 5              accumulated_portion
                iteration 1 : validation_acc = 93.12 %                  0.5
                iteration 2 : validation_acc = 95.02 %                  0.75
                iteration 3 : validation_acc = 95.78 %                  0.875
                iteration 4 : validation_acc = 96.54 %                  1.0
            
```
