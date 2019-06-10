import tensorflow as tf
import numpy as np
import os

#---------------------------------------------------------------------------------------------------

def save_4d_array(array , file_name):
    #file.write("array shape = " + str(array.shape))
    #file.write("\n")
    #file.write("[")
    file = open(file_name , 'a')
    for d1 in array:
     #   file.write("[")
        for d2 in d1:
      #      file.write("[")
            for d3 in d2:
       #         file.write("[")
                np.savetxt(file, d3, fmt='%s')
        #        file.write("]")
         #   file.write("]")
        #file.write("]")
    #file.write("]")
    file.write("\n")
    file.close()
    return

#-----------------------------------------------------------------------------------------------------

def save_1d_array(array , file_name):
    #file.write("array shape = " + str(array.shape))
    #file.write("\n")
    #file.write("[")
    file = open(file_name , "a")
    np.savetxt(file , array , fmt='%s')
    #file.write("]")
    #file.write("\n")
    return

#---------------------------------------------------------------------------------------------------

def extract_number_from_str(str):
    number = ''
    for let in str:
        if(ord(let) < 48 or ord(let) > 57):
            continue
        else:
            number += let
    return int(number)

#------------------------------------------------------------------------------------------------------

def sort(keys , num_layer):
    newKeys = list()
    for i in range(num_layer):
        newKeys.append(0)

    for key in keys:
        if('kernel' not in key):
            continue
        else:
            index = extract_number_from_str(key)
            newKeys[index - 1] = key

    return newKeys

#-----------------------------------------------------------------------------------------------------

def check_exist_path(path):
    if(not os.path.exists(path)):
        os.mkdir(path)
        print("directory {} was created".format(path))
        return
    else:
        return

def save_weights_to_file(var_list , keys , num_layer=34 , save='all' , file_path='./'):
    if(save == 'convolutional'):
        keys = sort(keys , num_layer=num_layer)
   # for key in keys:
   #     print(key)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

      #  tensor = var_list[keys[0]]

        print("start saving...")

       # numpy = tensor.eval()
        check_exist_path(file_path)
        info_file_name = '/info.txt'
        out_file = open(file_path + info_file_name , 'a')

        out_file.write("------------------------------------weights for resnet-34--------------------------------------")
        file_name_prefix = file_path + '/convolutional'
        for i in range(len(keys)):

            t = var_list[keys[i]]
            
            out_file.write("\n")

            out_file.write(str(keys[i]))

            out_file.write("\n")

            npp = t.eval()
            file_name = file_name_prefix + str(i + 1) + ".txt"
            print(file_name)

            if(len(npp.shape) == 4):
                if(save == 'all' or save == 'convolutional'):
                    print("saving 4d array...")
                    save_4d_array(npp , file_name)

            elif(len(npp.shape) == 1):
                if(save == 'all' or save == 'no-conv'):
                    save_1d_array(npp , out_file)
                    
                else:
                    continue

            print("array with shape : " + str(npp.shape) + " " + "was saved successfully")

        out_file.close()

        print("saving operation : done :)")

#-----------------------------------------------------------------------------------------------------