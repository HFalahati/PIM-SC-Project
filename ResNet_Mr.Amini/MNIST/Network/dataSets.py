from tensorflow.examples.tutorials.mnist import input_data


#-------------------------------------------------------------------------------------------

def MNIST_dataset():

    mnist = input_data.read_data_sets("./MNIST_dataset", one_hot=True)
    mnist_dataset = dict()
    mnist_dataset["train_images"] , mnist_dataset["train_labels"] = mnist.train.images[:50000] , mnist.train.labels[:50000]
    mnist_dataset["validation_images"] , mnist_dataset["validation_labels"] = mnist.train.images[50000:] , mnist.train.labels[50000:]
    mnist_dataset["test_images"] , mnist_dataset["test_labels"] = mnist.test.images , mnist.test.labels
    return mnist_dataset

#--------------------------------------------------------------------------------------------

'''
TODO : other datasets

'''
