import tensorflow as tf

#---------------------------------------------------------------------------------------

def create_weights(depth=32 , num_classes=10):
    weights = dict()
    kernel_size = 3
    num_kernel = 16
    kernel_depth = 1
    layer_name_prefix = "convolution"
    layer_counter = 1
    num_stacks = 3
    num_layer_per_stack = (depth - 2) // 3
    weights[layer_name_prefix + str(layer_counter)] = tf.Variable(tf.random_normal([kernel_size , kernel_size , kernel_depth , num_kernel]) , name=layer_name_prefix + str(layer_counter))
    for stack in range(num_stacks):
        for layer in range(num_layer_per_stack):
            if(stack == 0):
                kernel_depth = num_kernel
            layer_counter += 1
            weights[layer_name_prefix + str(layer_counter)] = tf.Variable(tf.random_normal([kernel_size , kernel_size , kernel_depth , num_kernel]) , name=layer_name_prefix + str(layer_counter))
            kernel_depth = num_kernel
        num_kernel *= 2

    num_kernel /= 2
    weights["fc"] = tf.Variable(tf.random_normal([4*4*int(num_kernel) , num_classes]) , name="fc_layer")

    return weights

#--------------------------------------------------------------------------------------------------

def get_weights_info(weights):
    for layer_name in weights.keys():
        print(layer_name , weights[layer_name])
    return

#---------------------------------------------------------------------------------------------