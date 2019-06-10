import tensorflow as tf
import numpy as np
import os

#-----------------------------------------------------------------------------------------

def create_dir(dir_name):
    dir = os.path.join(os.getcwd(), dir_name)
    if not os.path.isdir(dir):
        os.makedirs(dir)
   # print("{} directory was created successfully.".format(dir_name))
    return

# ---------------------------------------------------------------------------------------------------

def save_4d_array(array, file_name):
    # file.write("array shape = " + str(array.shape))
    # file.write("\n")
    # file.write("[")
    file = open(file_name, 'a')
    for d1 in array:
        #   file.write("[")
        for d2 in d1:
            #      file.write("[")
            for d3 in d2:
                #         file.write("[")
                np.savetxt(file, d3, fmt='%s')
                #        file.write("]")
                #   file.write("]")
                # file.write("]")
    # file.write("]")
    file.close()
    return


# -----------------------------------------------------------------------------------------------------

def save_1d_array(array, file_name):
    # file.write("array shape = " + str(array.shape))
    # file.write("\n")
    # file.write("[")
    file = open(file_name, "a")
    np.savetxt(file, array, fmt='%s')
    # file.write("]")
    # file.write("\n")
    file.close()
    return

#---------------------------------------------------------------------------------------

def save_2d_array(array , file_name):
    file = open(file_name , "a")
    for d1 in array:
        np.savetxt(file , d1 , fmt='%s')
    file.close()
    return

#-------------------------------------------------------------------------------------------

def saveLog(loss , acc , iteration , dir , file_name):
    file = open(dir + file_name , "a")
    record = str(loss) + " " + str(acc) + " " + str(iteration) + "\n"
    file.write(record)
    file.close()
    return

# -----------------------------------------------------------------------------------------------------

def save_networkInfo(model , n_epoch , batch_size , optimizer , dataset , testing_acc):
    file_name = "NetworkInfo.txt"
    file = open(file_name , "w")
    record = "model : " + str(model) + "\n" \
             "n_epoch : " + str(n_epoch) + "\n" \
             "batch_size : " + str(batch_size) + "\n" \
             "optimizer : " + str(optimizer) + "\n" \
             "dataset : " + str(dataset) + "\n" \
             "testing accuracy : " + str(testing_acc) + "\n"
    file.write(record)
    file.close()
    return

#---------------------------------------------------------------------------------------------

def save_weights_to_file(session , weights , weights_dir , dir):
        working_dir = weights_dir + dir
        create_dir(working_dir)
        for name in weights.keys():
            file_name = working_dir + "/" + str(name) + ".txt"
            weight = session.run(weights[name])
            if (len(weight.shape) == 4):
                save_4d_array(weight , file_name)

            elif (len(weight.shape) == 1):
                    save_2d_array(weight , file_name)

            else:
                    continue

            #print("array with shape : " + str(weight.shape) + " " + "was saved successfully")

        #print("saving operation : done :)")

# -----------------------------------------------------------------------------------------------------