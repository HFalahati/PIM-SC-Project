import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from Saver import saver
from NetworkWeights import create_weights , get_weights_info
#-----------------------------------------------------------------------------------

mnist = input_data.read_data_sets("./MNIST_dataset" , one_hot=True)

#-----------------------------------------------------------------------------------

tf.reset_default_graph()

#Hyperparameters
model="ResNet32"
opt_type="AdamOptimizer"
dataset="MNIST"
n_epoch = 3
batch_size = 128
display_step = 100
momentum = 0.9
weight_decay = 1e-4
depth = 32
num_input = 28*28
num_classes = 10
#learning_rate = 0.001

#-------------------------------------------------------------------------------------

weights_dir = "MNIST_weights"
log_dir = "log"
validation_log_file = "/validation_log.txt"
training_log_file = "/training_log.txt"

#-------------------------------------------------------------------------------------

#inputs

Input = tf.placeholder(dtype=tf.float32 , shape=(None , num_input) , name="inputs")
Label = tf.placeholder(dtype=tf.int32 , shape=(None , num_classes) , name="labels")
learning_rate = tf.placeholder(dtype=tf.float32 , name="learning_rate")

#--------------------------------------------------------------------------------------

weights = create_weights()
weights_info = get_weights_info(weights)

#--------------------------------------------------------------------------------------------

def reshape(x , batch_size=-1 , height=28 , width=28 , num_channel=1):
    return tf.reshape(x , [batch_size , height , width , num_channel])

#-----------------------------------------------------------------------------------------------

def first_layer(input , layer_name , stride=1):
    global weights
    x = tf.nn.conv2d(input , weights[layer_name] , strides=[1 , stride , stride , 1] , padding="SAME")
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.nn.relu(x)

#-------------------------------------------------------------------------------------------

def ResBlock(input ,layer_names , stride=1):
    global weights
    weight1 = weights[layer_names[0]]
    weight2 = weights[layer_names[1]]
    x = tf.nn.conv2d(input , weight1 , strides=[1 , stride , stride , 1] , padding="SAME")
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.nn.conv2d(x , weight2 , strides=[1 , 1 , 1 , 1] , padding="SAME")
    x = tf.keras.layers.BatchNormalization()(x)
    return x

#------------------------------------------------------------------------------------------

def get_layer_names(counter):
    prefix = "convolution"
    names = list()
    counter += 1
    names.append(prefix + str(counter))
    counter += 1
    names.append(prefix + str(counter))
    return counter , names

#----------------------------------------------------------------------------------------------

def avg_pool(x , stride=2 , k=2):
    return tf.nn.avg_pool(x , ksize=[1 , k , k , 1] , strides=[1 , stride , stride , 1] , padding="SAME")

#---------------------------------------------------------------------------------------------

def fully_layer(x):
    global weights
    fc = tf.reshape(x , [-1 , weights["fc"].get_shape().as_list()[0]])
    fc = tf.matmul(fc , weights["fc"])
    return fc

#-----------------------------------------------------------------------------------------------

def ResNet_v1(input):
    global weights , depth
    x = reshape(input)
    layer_name_prefix = "convolution"
    layer_counter = 1
    x = first_layer(input=x , layer_name=layer_name_prefix + str(layer_counter))
    num_stacks = 3
    num_resblock_per_stack = (depth - 2) // 6
    for stack in range(num_stacks):
        for block in range(num_resblock_per_stack):
            stride = 1
            layer_names = list()
            if(stack > 0 and block == 0):
                stride = 2
            layer_counter , layer_names = get_layer_names(layer_counter)
            y = ResBlock(input=x , layer_names=layer_names , stride=stride)
            try:
                tf.add(x , y)
            except:
                x = y

    x = avg_pool(x)
    x = fully_layer(x)
    return x

#-----------------------------------------------------------------------------------------------

def lr_sceduler(epoch):
    Learning_rate = 1e-3
    if(epoch > 80):
        Learning_rate = 3.125e-05
    elif(epoch > 60):
        Learning_rate = 6.25e-05
    elif(epoch > 40):
        Learning_rate = 0.000125
    elif(epoch > 20):
        Learning_rate = 0.00025
    elif(epoch > 10):
        Learning_rate = 0.0005
    return Learning_rate

#---------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------

'''
def run_weights(session):
    global weights
    print("start saving weights...")
    for layer_name in weights.keys():
        layer_weights = session.run(weights[layer_name])
        if(layer_name == "fc"):
            saver.save_2D_array(layer_weights , layer_name + ".txt")
        else:
            saver.save_4d_array(layer_weights , layer_name + ".txt")
    print("saving weights done.")

    return
'''
#---------------------------------------------------------------------------------------------

def regularizer(weights , weight_decay):
    regularize = 0
    for layer_name in weights.keys():
        regularize += tf.nn.l2_loss(weights[layer_name])

    return weight_decay*regularize

#----------------------------------------------------------------------------------------------

def clip_weights():
    global weights
    min_clip = tf.constant(-0.95 , dtype=tf.float32 , shape=())
    max_clip = tf.constant(0.95 , dtype=tf.float32 , shape=())
    for name in weights.keys():
        weights[name] = tf.clip_by_value(weights[name] , min_clip , max_clip)

#---------------------------------------------------------------------------------------

saver.create_dir(weights_dir)
saver.create_dir(log_dir)

logits = ResNet_v1(input=Input)
prediction = tf.nn.softmax(logits=logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits , labels=Label))
loss_op = tf.reduce_mean(loss + regularizer(weights , weight_decay))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss=loss_op)


correct_pred = tf.equal(tf.argmax(prediction , 1) , tf.argmax(Label , 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred , tf.float32))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    Learning_rate = 1e-03
    old_acc = 0
    n_batches = len(mnist.train.images) // batch_size
    for epoch in range(n_epoch):
        print("learning rate : {}".format(Learning_rate))
        for batch in range(n_batches):
            batch_x , batch_y = mnist.train.next_batch(batch_size)
            _ = sess.run(train_op , feed_dict={Input:batch_x , Label:batch_y , learning_rate:Learning_rate })

            if(epoch*n_batches + batch == 0 or (epoch*n_batches + batch) % display_step == 0):
                loss , acc = sess.run([loss_op , accuracy] , feed_dict={Input:batch_x , Label:batch_y})
                saver.saveLog(loss=loss , acc=acc , iteration=epoch*n_batches+batch , dir=log_dir , file_name=training_log_file)
                val_loss , val_acc = sess.run([loss_op , accuracy] , feed_dict={Input:mnist.test.images[:1000] , Label:mnist.test.labels[0:1000]})
                print("epoch : {} , mini_batch : {} , mini_batch loss : {:.4f} , training accuracy : {:.4f} , validation accuracy : {:.4f}".format(epoch , batch , loss , acc , val_acc))
                saver.saveLog(loss=val_loss , acc=val_acc , iteration=epoch*n_batches+batch , dir=log_dir , file_name=validation_log_file)
                clip_weights()
                if(val_acc > old_acc):
                    print("validation accuracy improved from {} to {}.".format(old_acc , val_acc))
                    old_acc = val_acc
                    dir = "/resnet{}_weights_val_{}".format(depth , val_acc)
                    #print("saving weights...")
                    saver.save_weights_to_file(session=sess , weights=weights , weights_dir=weights_dir , dir=dir)
                    #print("saving weights : done.")
        Learning_rate = lr_sceduler(epoch)

    print("optimization finished!")
    test_acc = sess.run(accuracy , feed_dict={Input:mnist.test.images[1000:] , Label:mnist.test.labels[1000:]})
    print("test accuracy : {:.4f}".format(test_acc))
    saver.save_networkInfo(model=model , n_epoch=n_epoch , batch_size=batch_size , optimizer=opt_type , dataset=dataset , testing_acc=test_acc)