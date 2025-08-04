import numpy as np
from polygonscraper import PolygonData
from kernelnorm import conv_train

api_key = 'aSE_Q2Vmnh9BCPSumGUceCJphILlB3ag'
ticker = 'AAPL'
limit = 20000

min_data = PolygonData(api_key).retrieve_data(ticker, limit)

#Train
train_obj = conv_train(min_data, 7)
train_obj.set_conv(5000)
feature_map = train_obj.features()
weights = train_obj.top_filters(top_k=15)

#Test
test_obj = conv_train(min_data, 7)
test_obj.my_weights(weights)
test_obj.features()
trans_matrix = test_obj.trans_matrix()

print(trans_matrix)
test_obj.filter_plot2d()






