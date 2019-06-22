import tensorflow as tf
from config import weights_dir , depth , num_classes
import os
import numpy as np

#---------------------------------------------------------------------------------------


def create_weights(depth=32 , num_classes=10 , initial_values=None):
    weights = dict()
    kernel_size = 3
    num_kernel = 16
    kernel_depth = 1
    layer_name_prefix = "convolution"
    layer_counter = 1
    num_stacks = 3
    num_layer_per_stack = (depth - 2) // 3
    if(initial_values == None):
        weights[layer_name_prefix + str(layer_counter)] = tf.Variable(tf.random_normal([kernel_size , kernel_size , kernel_depth , num_kernel]) , name=layer_name_prefix + str(layer_counter))
    else:
        weights[layer_name_prefix + str(layer_counter)] = initial_values[layer_name_prefix + str(layer_counter)]
    for stack in range(num_stacks):
        for layer in range(num_layer_per_stack):
            if(stack == 0):
                kernel_depth = num_kernel
            layer_counter += 1
            if(initial_values == None):
                weights[layer_name_prefix + str(layer_counter)] = tf.Variable(tf.random_normal([kernel_size , kernel_size , kernel_depth , num_kernel]) , name=layer_name_prefix + str(layer_counter))
            else:
                weights[layer_name_prefix + str(layer_counter)] = initial_values[layer_name_prefix + str(layer_counter)]
            kernel_depth = num_kernel
        num_kernel *= 2

    num_kernel /= 2
    if(initial_values == None):
        weights["fc"] = tf.Variable(tf.random_normal([4*4*int(num_kernel) , num_classes]) , name="fc_layer")
    else:
        weights["fc"] = initial_values["fc"]

    return weights

#--------------------------------------------------------------------------------------------------

def get_weights_info(weights):
    for layer_name in weights.keys():
        print(layer_name , weights[layer_name])
    return

#---------------------------------------------------------------------------------------------

def get_weights_directory(dir=weights_dir):
    parent_directory = os.path.join(os.getcwd() , dir)
    return ([x[0] for x in os.walk(parent_directory)][-1])

#-----------------------------------------------------------------------------------------------

def extract_number_from_str(str):
    number = ''
    for let in str:
        if(ord(let) < 48 or ord(let) > 57):
            continue
        else:
            number += let
    return int(number)

#------------------------------------------------------------------------------------------------

def sort(files_name):
    new_files = [x for x in range(len(files_name))]
    for name in files_name:
        if("fc" not in name):
            index = extract_number_from_str(name)
            new_files[index - 1] = name
        else:
            new_files[-1] = name
    return new_files

#----------------------------------------------------------------------------------------------

def get_weights_files(weights_directory):
    return sort([files for files in os.listdir(weights_directory) if os.path.isfile(os.path.join(weights_directory, files))])

#-------------------------------------------------------------------------------------------------

def get_shapes(weights):
    shapes = list()
    for layer_name in weights.keys():
        shapes.append(weights[layer_name].get_shape().as_list())
    return shapes

#---------------------------------------------------------------------------------------------------

def reshape_selector(selector):
    shapes = get_shapes(create_weights(depth=depth , num_classes=num_classes))
    new_selector = dict()
    layer_name_prefix = "convolution"
    for layer_counter in range(len(shapes)):
        if (layer_counter == len(shapes) - 1):
            new_selector["fc"] = tf.convert_to_tensor(selector["fc"].reshape(shapes[layer_counter]), dtype=tf.float32)
        else:
            new_selector[layer_name_prefix + str(layer_counter + 1)] = tf.convert_to_tensor(selector[layer_name_prefix + str(layer_counter + 1)].reshape(shapes[layer_counter]),
                                                                              dtype=tf.float32)

    return new_selector

#--------------------------------------------------------------------------------------------------



def load_weights(weights_dir):
    working_directory = os.path.join(os.getcwd() , weights_dir)
    print(get_weights_directory(working_directory))
    files_name = get_weights_files(get_weights_directory(working_directory))
    shapes = get_shapes(create_weights(depth=depth , num_classes=num_classes))
    layer_name_prefix = "convolution"
    weights = dict()
    for layer_counter in range(len(files_name)):
        file_name = os.path.join(get_weights_directory(working_directory), str(files_name[layer_counter]))
        array = np.loadtxt(file_name)
        if(layer_counter == len(files_name) - 1):
            weights["fc"] = tf.Variable(array.reshape(shapes[layer_counter]) , dtype=tf.float32)
        else:
            weights[layer_name_prefix + str(layer_counter + 1)] = tf.Variable(array.reshape(shapes[layer_counter]) , dtype=tf.float32)

    return weights

#--------------------------------------------------------------------------------------------------

def load_weights_for_quantization(weights_dir):
    working_directory = os.path.join(os.getcwd(), weights_dir)
    files_name = get_weights_files(get_weights_directory(working_directory))
    layer_name_prefix = "convolution"
    weights = dict()
    for layer_counter in range(len(files_name)):
        file_name = os.path.join(get_weights_directory(working_directory), str(files_name[layer_counter]))
        if (layer_counter == len(files_name) - 1):
           weights["fc"] =  np.loadtxt(file_name)
        else:
           weights[layer_name_prefix + str(layer_counter + 1)] = np.loadtxt(file_name)

    return weights
