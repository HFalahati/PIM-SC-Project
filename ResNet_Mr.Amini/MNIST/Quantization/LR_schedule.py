def MNIST_lr_sceduler(epoch):
    Learning_rate = 1e-3
    if(epoch > 80):
        Learning_rate = 1e-8
    elif(epoch > 70):
        Learning_rate = 1e-7
    elif(epoch > 60):
        Learning_rate = 1e-6
    elif(epoch > 50):
        Learning_rate = 1e-5
    elif(epoch > 30):
        Learning_rate = 1e-4
    return Learning_rate

def MNIST_INQ_lr_scheduler(acc):
    Learning_rate = 1e-3
    if(acc > 0.99):
        Learning_rate = 1e-5
    elif(acc > 0.97):
        Learning_rate = 1e-4

    return Learning_rate

'''
TODO : other datasets lr_scheduler

'''