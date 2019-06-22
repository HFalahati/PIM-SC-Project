import tensorflow as tf
from dataSets import MNIST_dataset
from LR_schedule import MNIST_lr_sceduler , MNIST_INQ_lr_scheduler
import numpy as np
from Saver import saver
from NetworkWeights import create_weights , get_weights_info , load_weights , reshape_selector
from config import *
from Optimizer.INQOptimizer import INQ_Optimizer

#-----------------------------------------------------------------------------------
weights = dict()
#-------------------------------------------------------------------------------------------

def load_dataset(dataset="MNIST"):
    if(dataset == "MNIST"):
        return MNIST_dataset()
    '''
    TODO : 
    else:
        <load other datasets>
    '''

#--------------------------------------------------------------------------------------------

def Next_batch(batch_size , num_example = 50000):
    indexes = np.arange(num_example)
    np.random.shuffle(indexes)
    return indexes[:batch_size]

#----------------------------------------------------------------------------------------------

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

def regularizer(weights , weight_decay):
    regularize = 0
    for layer_name in weights.keys():
        regularize += tf.nn.l2_loss(weights[layer_name])

    return weight_decay*regularize

#----------------------------------------------------------------------------------------------

def clip_weights():
    global weights
    min_clip = tf.constant(-0.99999999 , dtype=tf.float32 , shape=())
    max_clip = tf.constant(0.99999999 , dtype=tf.float32 , shape=())
    for name in weights.keys():
        weights[name] = tf.clip_by_value(weights[name] , min_clip , max_clip)

#---------------------------------------------------------------------------------------------

def to_list():
    global weights
    params = list()
    for name in weights.keys():
        params.append(weights[name])
    return params
#---------------------------------------------------------------------------------------

def apply_sel(selector , grad):
    counter = 0
    for layer_name in selector.keys():
        grad[counter] = (tf.multiply(selector[layer_name] , grad[counter][0]) , grad[counter][1])
        counter += 1
    return grad

def create_graph(selector=None):
    global weights
    tf.reset_default_graph()
    selector = reshape_selector(selector)
    # -------------------------------------------------------------------------------------
    # inputs

    Input = tf.placeholder(dtype=tf.float32, shape=(None, num_input), name="inputs")
    Label = tf.placeholder(dtype=tf.int32, shape=(None, num_classes), name="labels")
    learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

    # --------------------------------------------------------------------------------------
    # weights = create_weights(depth=depth , num_classes=num_classes)
    weights = load_weights(weights_dir=quantized_weights_dir)

    # create graph
    logits = ResNet_v1(input=Input)
    prediction = tf.nn.softmax(logits=logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Label))
    loss_op = tf.reduce_mean(loss + regularizer(weights, weight_decay))

    optimizer = INQ_Optimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss=loss_op, var_list=to_list())
    new_grads_and_vars = apply_sel(selector , grads_and_vars)
    train_op = optimizer.apply_gradients(grads_and_vars=new_grads_and_vars)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # ------------------------------------------------------------------------------------------------

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    weights_info = get_weights_info(weights)

    return (Input , Label , train_op , accuracy , loss_op , config , init , learning_rate)
#create nedded directories
saver.create_dir(weights_dir)
saver.create_dir(log_dir)

#-----------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------

def Train(selector , iteration):
    global weights
    Input , Label , train_op , accuracy , loss_op , config , init , learning_rate = create_graph(selector)
    dataset = load_dataset(dataset="MNIST")
    with tf.Session(config=config) as sess:
        sess.run(init)
        Learning_rate = 1e-03
        val_acc = 0
        n_batches = len(dataset["train_images"]) // batch_size
        for epoch in range(n_epoch):
            print("learning rate : {}".format(Learning_rate))
            for batch in range(n_batches):
                batch_indexes = Next_batch(batch_size)
                batch_x , batch_y = dataset["train_images"][batch_indexes] , dataset["train_labels"][batch_indexes]
                _ = sess.run(train_op , feed_dict={Input:batch_x , Label:batch_y , learning_rate:Learning_rate })

                if(epoch*n_batches + batch == 0 or (epoch*n_batches + batch) % display_step == 0):
                    loss , acc = sess.run([loss_op , accuracy] , feed_dict={Input:batch_x , Label:batch_y})
                    saver.saveLog(loss=loss , acc=acc , iteration=epoch*n_batches+batch , dir=log_dir , learning_rate=Learning_rate , file_name=training_log_file)
                    val_loss , val_acc = sess.run([loss_op , accuracy] , feed_dict={Input:dataset["validation_images"] , Label:dataset["validation_labels"]})
                    print("epoch : {} , mini_batch : {} , mini_batch loss : {:.4f} , training accuracy : {:.4f} , validation accuracy : {:.4f}".format(epoch , batch , loss , acc , val_acc))
                    saver.saveLog(loss=val_loss , acc=val_acc , iteration=epoch*n_batches+batch , dir=log_dir , learning_rate=Learning_rate , file_name=validation_log_file)
                #    if(val_acc > old_acc):
                #        print("validation accuracy improved from {:.4f} to {:.4f}.".format(old_acc , val_acc))
                #        old_acc = val_acc
                #        dir = "/resnet{}_weights_val_{:.4f}".format(depth , val_acc)
                #        #print("saving weights...")
                #        saver.save_weights_to_file(session=sess , weights=weights , weights_dir=weights_dir , dir=dir)
                        #print("saving weights : done.")

            Learning_rate = MNIST_INQ_lr_scheduler(val_acc)
        print("optimization finished!")
        dir = "/resnet{}_weights_Z_iteration{}".format(depth , iteration)
        saver.save_weights_to_file(session=sess, weights=weights, weights_dir=weights_dir, dir=dir)
        test_acc = sess.run(accuracy , feed_dict={Input:dataset["test_images"] , Label:dataset["test_labels"]})
        print("test accuracy : {:.4f}".format(test_acc))
        #saver.save_networkInfo(model=model , n_epoch=n_epoch , batch_size=batch_size , optimizer=opt_type , dataset=Dataset , testing_acc=test_acc)


#-------------------------------------------------------------------------------------------------------

def Inference(selector):
    global weights
    Input , Label , _ , accuracy , __ , config , init , ___ = create_graph(selector)
    dataset = load_dataset(dataset="MNIST")
    with tf.Session(config=config) as sess:
        sess.run(init)
        test_acc = sess.run(accuracy, feed_dict={Input: dataset["test_images"], Label: dataset["test_labels"]})
        validation_acc = sess.run(accuracy , feed_dict={Input: dataset["validation_images"], Label: dataset["validation_labels"]})
      #  saver.save_networkInfo(model=model, n_epoch=n_epoch, batch_size=batch_size, optimizer=opt_type, dataset=dataset,
      #                         testing_acc=test_acc , validation_acc= validation_acc)
        print("test accuracy : {:.4f} , validation accuracy : {:.4f}".format(test_acc , validation_acc))

#---------------------------------------------------------------------------------------------------------
