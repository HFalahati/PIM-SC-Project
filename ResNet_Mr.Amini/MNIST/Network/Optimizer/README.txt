import tensorflow as tf
.
.
.
.
.
#construct your Network here
.
.
.
.

#-------------------------------------------------------------------------

logits = network(<input> , <weights>)
prediction = tf.nn.softmax(logits)

#------------------------------------------------------------------------------

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits , labels=<Labels>))

#-----------------------------------------------------------------------------------

#create your Optimizer! : 
optimizer = INQ_Optimizer(learning_rate=<learning_rate>)

#compute gradients : 
grad_and_vars = optimizer.compute_gradients(loss=loss_op , var_list=<params>) # params is a list of Tensors (trainable weights in your network)

new_grad_and_vars = [(apply_sel(sel , gv[0]) , gv[1]) for gv in grads_and_vars] # sel is T matrix that created in quantization phase.

#---------------------------------------------------------------------

#apply_sel : 
def apply_sel(sel , grad);
	return tf.multiply(sel , grad)

#------------------------------------------------------------------------

#update weights : 
train_op = optimizer.apply_gradients(grads_and_vars=new_grads_and_vars) 

#------------------------------------------------------------------------

.
.
.
.
.
.
.
.
.

#---------------------------------------------------------------------------

#run the network : 
_ = sess.run(train_op , feed_dict={input:<input> , labels:<labels>})

#--------------------------------------------------------------------------