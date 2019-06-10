import numpy as np

#----------------------------------------------------------------------------------------------

accumulated_portion = [50 , 75 , 87.5 , 100]

#-----------------------------------------------------------------------------------------------

def read_weights(file_name):
    weights = np.loadtxt(file_name)
    return weights

#------------------------------------------------------------------------------------------------

def Initialize(num_layer , data_directory):
    A1 = dict()
    A2 = dict()
    T = dict()
    layer_name_prefix = 'convolutional'
    for layer in range(num_layer):
        A1[layer_name_prefix + str(layer + 1)] = np.array([] , dtype=np.float32)
        print("loading weights from layer " + str(layer + 1) + " ...")
        A2[layer_name_prefix + str(layer + 1)] = read_weights(data_directory + layer_name_prefix + str(layer + 1) + ".txt")
        print("loading weights from layer " + str(layer + 1) + " done :)")
        T[layer_name_prefix + str(layer + 1)] = np.ones(len(A2[layer_name_prefix + str(layer + 1)]) , dtype=np.float32)
    return (A1 , A2 , T)

#---------------------------------------------------------------------------------------------

#weights = read_weights('../test.txt')

#print(Initialize(weights))

#-----------------------------------------------------------------------------------------------

def select_weights(num_weights , all=False):
    index = (np.arange(num_weights))
    if(all):
        selected_weights = index
        unselected = np.array([])
    else:
        np.random.shuffle(index)
        selected_weights = index[:num_weights // 2 + 1]
        unselected = index[num_weights // 2 + 1:]
    return (selected_weights,unselected)

#--------------------------------------------------------------------------------------------

def clc_s(A1):
    return max(max(A1) , abs(min(A1)))

#-----------------------------------------------------------------------------------------
#print(clc_s(weights))
#-------------------------------------------------------------------------------------------

def clc_n1(s):
    return np.floor(np.log2(4*s/3))

#-------------------------------------------------------------------------------------------
#print(clc_n1(clc_s(weights)))
#------------------------------------------------------------------------------------------

def clc_n2(n1 , b):
    return (n1 + 1 - (2**(b-1) / 2))

#-----------------------------------------------------------------------------------------
#print(clc_n2(clc_n1(clc_s(weights)) , 5))
#------------------------------------------------------------------------------------------

def create_P_set(n1 , n2):
    P = list()
    i = n2
    while(i != n1 + 1):
        P.append(2**i)
        P.append(-2**i)
        i += 1
    P.append(0)
    P.sort()
    return P

#--------------------------------------------------------------------------------------------
#n1 = clc_n1(clc_s(weights))
#n2 = clc_n2(n1 , 5)
#print(create_P_set(n1 , n2))
#---------------------------------------------------------------------------------------------
def sgn(x):
    if(x < 0):
        return -1
    elif(x > 0):
        return 1
    else:
        return 0

#-----------------------------------------------------------------------------------------------

def do_quantization(weights, P):
    for i in range(len(weights)):
            quantized = False
            for j in range(len(P) - 1):
                alpha = P[j]
                beta = P[j + 1]
                if((alpha + beta)/2 <= abs(weights[i]) and abs(weights[i]) < 3*beta/2):
                    weights[i] = beta * sgn(weights[i])
                    quantized = True
                    break

            if(not quantized):
                weights[i] = 0
    return weights

#-----------------------------------------------------------------------------------------------

#print(do_quantization(weights , create_P_set(n1 , n2)))

#-----------------------------------------------------------------------------------------------

def main(num_layer , bit_width , data_directory):
    print("loading weights....")
    A1 , A2 ,T = Initialize(num_layer , data_directory)
    print("loading weights done :))")
    layer_name_prefix = 'convolutional'
    for i in range(4):
        for layer in range(num_layer):
            if(i == 3):
                selected , unselected = select_weights(len(A2[layer_name_prefix + str(layer + 1)]) , all=True)
            else:
                selected, unselected = select_weights(len(A2[layer_name_prefix + str(layer + 1)]))
            A1[layer_name_prefix + str(layer + 1)] = A2[layer_name_prefix + str(layer + 1)][selected]
            if(i != 3):
                A2[layer_name_prefix + str(layer + 1)] = A2[layer_name_prefix + str(layer + 1)][unselected]
            else:
                A2[layer_name_prefix + str(layer + 1)] = np.array([])
            T[layer_name_prefix + str(layer + 1)][selected] = 0
            n1 = clc_n1(clc_s(A1[layer_name_prefix + str(layer + 1)]))
            n2 = clc_n2(n1 , bit_width)
            P = create_P_set(n1 , n2)
            A1[layer_name_prefix +  str(layer + 1)] = do_quantization(A1[layer_name_prefix + str(layer + 1)] , P)

        #just for test
        print(len(A2['convolutional10']))

    #just for test
    return (A1['convolutional3'])


num_laley = 34
bit_width = 5
data_directory = 'weights/'

conv3 = main(num_laley , bit_width , data_directory)
result = 'test.txt'
np.savetxt(result , conv3 , fmt='%5s')