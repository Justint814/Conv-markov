import numpy as np
from kernelnorm import conv_train

###  INITIALIZE DATA HERE ###
data = np.random.rand(500)

#Train
train_obj = conv_train(data, 8, stride=1) #  Make object to run 1D convolution over. (data, kernel_size, stride=stride).
train_obj.set_conv(30000) #  Initialize convolutional layer with N kernels (kernel_num).
feature_map = train_obj.features() #  Run convolution with initlialized settings and return feature map.
weights = train_obj.top_filters(top_k=15) #  Return top weights of top filters. (top_k=number of top filters).

#Test
test_obj = conv_train(data, 8, stride=8) # Initialize testing object with kernel size of 7. (data, kernel_size, stride=stride).
test_obj.my_weights(weights) # Input custom weights. In this case we use the top weights returned from [train_obj.top_filters(top_k=15)].
test_obj.features() # Initialize activations of each filter, i.e: Run conv1D and initialize feature map in conv_tran object.
trans_matrix = test_obj.trans_matrix() # Get transition matrix.
print(trans_matrix) #  Print the transition matrix
test_obj.filter_plot2d() # Plot each kernel and it's weights
test_obj.markov_chain() #  Draw markov chain.






