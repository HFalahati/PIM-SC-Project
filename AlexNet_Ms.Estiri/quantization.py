import numpy as np


iterate = '1'


def read_weights(file_name):
    weights = np.loadtxt(file_name)
    return weights


def generate_seed(size, s):
    np.random.seed(s)
    rand = np.arange(size)
    np.random.shuffle(rand)
    return rand


def generate_A1_A2_T(weights):
    seed = generate_seed(len(weights), 0)
    a = (len(weights)) // 2 + 1
    A1 = weights[seed[:a]]
    A2 = weights[seed[a:]]
    T = np.ones(len(weights))
    T[seed[:a]] = np.zeros(a)

    return A1, A2, T

# w = np.array([1,2,3,4,5,67,8,9,2,32,23,2,12])
# s = generate_seed(len(w), 0)
# generate_A1_A2_T(w, s)


def generate_n1_n2(weights, bit_width):
    s = max(abs(weights))
    n1 = np.floor(np.log2(4*s/3))
    n2 = (n1 + 1 - (2**(bit_width-2)))

    return n1, n2

def generate_P(n1, n2):
    result = [0]

    for i in range(int(n2), int(n1+1)):
        result.append(2**i)
        result.append(-2**i)

    result.sort()
    return result


def quantization(A1, P):

    for i in range(len(A1)):
        flag = False
        for j in range(len(P)-1):
            alpha = P[j]
            beta = P[j+1]
            if (alpha + beta) / 2 <= abs(A1[i]) and abs(A1[i]) < 3 * beta / 2:
                A1[i] = beta * np.sign(A1[i])
                flag = True
                break
        if not flag:
            A1[i] = 0
    return A1


def run (layer_i):
    weights = read_weights('weights/weight' + str(layer_i) + '.txt')
    n1, n2 = generate_n1_n2(weights, 5)
    P = generate_P(n1, n2)
    A1, A2, T = generate_A1_A2_T(weights)

    new_A1 = quantization(A1, P)

    np.savetxt('result/new_' + iterate + 'A1_layer_' + str(layer_i) + '.txt', new_A1,  fmt='%5s')
    np.savetxt('result/' + iterate + 'T_' + str(layer_i) + '.txt',  T, fmt='%5s')


if __name__ == '__main__':
    run(8)

