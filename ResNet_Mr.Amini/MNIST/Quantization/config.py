
#Network parameters

n_epoch = 5
batch_size = 128
display_step = 100
momentum = 0.9
weight_decay = 5e-4
depth = 32
num_input = 28*28
num_classes = 10

#-----------------------------------------------------------------------------------------------------
#Log parameters

model="ResNet32"
opt_type="SGD with momentum"
Dataset="MNIST"

weights_dir = "MNIST_weights"
log_dir = "log"
validation_log_file = "/validation_log.txt"
training_log_file = "/training_log.txt"
afterClip_validation_log_file = "/afterClip_validation_log.txt"

#----------------------------------------------------------------------------------------------------
#quantization parameters

bit_width = 5
quantized_weights_dir = "MNIST_quantized_weights"
accumulated_portion = [0.5 , 0.75 , 0.875 , 1]