import numpy as np
from Saver import saver
from NetworkWeights import *
from config import *
from INQ_Resnet import Train , Inference

#----------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

def Initialize():
    A1 = load_weights_for_quantization(weights_dir)
    T = dict()
    Indexes = dict()
    num_weights = dict()
    Sum = dict()
    for name in A1.keys():
        num_weights[name] = len(A1[name])
        Indexes[name] = np.arange(len(A1[name]))
        T[name] = np.ones(len(A1[name]) , dtype=np.float32)
        Sum[name] = 0
    return (A1 , T , Indexes , num_weights , Sum)

#---------------------------------------------------------------------------------------------

#weights = read_weights('../test.txt')

#print(Initialize(weights))

#-----------------------------------------------------------------------------------------------

def select_weights(Indexes , num_weights , a , Sum):
    index = Indexes[Indexes >= 0]
    portion = len(Indexes) * a
    a = int(num_weights // (portion - Sum))
    np.random.shuffle(index)
    selected_weights = index[:(num_weights // a)]
    try:
        unselected = index[(num_weights // a):]
    except:
        unselected = list()
    return (selected_weights,len(unselected) , portion)


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

def do_quantization(weights, P , zero=False):
    for i in range(len(weights)):
        if(zero):
            weights[i] = 0
        else:
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

def main():
    print("loading weights....")
    A1 ,T , Indexes , num_weights , Sum = Initialize()
    print("loading weights done :))")
    for i in range(len(accumulated_portion)):
        for name in A1.keys():
            print("start {} layer quantization in iteration {}".format(name , i+1))
            selected , num_weights[name] , Sum[name] = select_weights(Indexes=Indexes[name] , num_weights=num_weights[name] , a=accumulated_portion[i] , Sum=Sum[name])
            Indexes[name][selected] = -1
            T[name][selected] = 0
            n1 = clc_n1(clc_s(A1[name]))
            n2 = clc_n2(n1 , bit_width)
            P = create_P_set(n1 , n2)
            if(n1 < -8):
                A1[name][selected] = do_quantization(A1[name][selected] , P , zero=True)
            else:
                A1[name][selected] = do_quantization(A1[name][selected], P)
       # print("start saving...")
        dir = "/resnet{}_quantized_weights_iteration{}".format(depth , i+1)
        with tf.Session() as sess:
            saver.save_weights_to_file(session=sess , weights=A1 , weights_dir=quantized_weights_dir , dir=dir , numpy_array=True)
        if(i < len(accumulated_portion)-1):
            Train(selector=T , iteration=i+1)
            A1 = load_weights_for_quantization(weights_dir)
        else:
            Inference(selector=T)
    return

if __name__ == '__main__':
    main()
