
#Network parameters

n_epoch = 100
batch_size = 128
display_step = 100
momentum = 0.9
weight_decay = 1e-4
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