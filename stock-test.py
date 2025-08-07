import numpy as np
from stockstream import PolygonData
from kernelnorm import conv_train


api_key = 'aSE_Q2Vmnh9BCPSumGUceCJphILlB3ag'
ticker = 'AAPL'
limit = 900000

min_data = PolygonData(api_key).retrieve_data(ticker, limit)
five_min = min_data[4:-1:3]

#Train
train_obj = conv_train(min_data, 8, stride=1)
train_obj.set_conv(30000)
feature_map = train_obj.features()
weights = train_obj.top_filters(top_k=15)

#Test
test_obj = conv_train(min_data, 8, stride=8) # Initialize testing object with k_size of 7
test_obj.my_weights(weights) # Input custom weights
test_obj.features() # Initialize activations of each filter
trans_matrix = test_obj.trans_matrix() # Get transition matrix
print(trans_matrix)
#test_obj.filter_plot2d() # Plot each kernel and it's weights
test_obj.markov_chain()






