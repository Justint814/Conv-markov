import numpy as np
import sys
import keras
import plotly.graph_objects  as go
from markovchain import MarkovChain
#Shape formate (outer, middle, inner)
#conv1D input shape (batch size, steps, channels) channels should be 1 for timeseries data. steps should be the amount of data in one time series. In this case, it should be the size of the kernel. Batch size should be the number of batches.


#Class used for collecting top filters 
class conv_train:
    def __init__(self, series, k_size, stride=1):
        self.series = series
        self.k_size = k_size
        self.stride = stride
        self.input = self.kernel_batch()

    
    #Helper function to apply min-max scaling to a numpy array
    def min_max(data):
        max = np.max(data)
        min = np.min(data)
        
        range = max - min

        if range > 0:
            return (data - min) / range
        else:
            return np.zeros(len(data))

    #Function to take time series data and convert it into a format for Keras Convlayer1D. (batches, steps, channels)
    def kernel_batch(self): #Inputs: D numpy array of values, kernel size
        length = np.shape(self.series)[0]
        batches = []

        for i in range(0, length - self.k_size, self.stride):
            stop = i + self.k_size
            target = self.series[i:stop]

            batch = conv_train.min_max(target)
            batches.append(batch)
        
        batches = np.expand_dims(np.array(batches), axis=-1)

        return batches

    def set_conv(self, filters, weight=True):
        self.conv_layer = keras.layers.Conv1D(filters, self.k_size, activation='relu')
        if weight==True:
            self.make_weights(self.k_size, filters)

            #Call conv layer to initalize weights
            self.conv_layer(self.input)

            #Apply target weights to filters
            self.conv_layer.set_weights([self.weight_arr, self.bias_arr])
            self.conv_layer.trainable = False

    #Function to set output and return feature map
    def features(self):
        self.output = self.conv_layer(self.input) #(batches, steps, filter activations)
        self.out_shape = np.shape(self.output)
        self.num_batches = self.out_shape[0]

        #Access weights and save to class variable
        self.weight_arr = np.array(self.conv_layer.get_weights()[0])

        return self.output
    
    #Function to sort output data by feature activation
    def top_filters(self, top_k=2):
        '''
        avg_activation = np.mean(self.output, axis=(0, 1))

        #Get indices of filters with highest average activation
        self.filter_indices = np.argsort(avg_activation)[-top_k:]

        #Get weight array of highest activation filters. Indices correspond to innermost dimension of weight array returned by conv.markov.weights()
        self.target_weights = self.weight_arr[:,:,self.filter_indices]
        '''
        top_f = []
        for i in range(self.num_batches):
            top_f.append(np.argmax(self.output[i,0,:]))

        top_f = np.array(top_f)
        indices, counts = np.unique(top_f, return_counts=True)
        weight_ind = np.row_stack((indices, counts))
        sort_ind = np.argsort(weight_ind[1,:])
        self.filter_indices = weight_ind[0,sort_ind][-top_k:]
        self.target_weights = self.weight_arr[:,:,self.filter_indices]


        self.num_filters = np.shape(self.target_weights)[2]
        return self.target_weights

    #Return weight array after obtaining feature map
    def weights(self):
        return self.weight_arr
    
    #Configure conv_layer to contain user input weights
    def my_weights(self, weights): #Weights shape: (k_size, 1, filters)
        self.num_filters = np.shape(weights)[2]
        self.weight_arr = weights
        self.bias_arr = np.zeros(self.num_filters)

        #Initialize conv layer
        self.conv_layer = keras.layers.Conv1D(self.num_filters, self.k_size, activation='relu')
        #Call conv layer to initalize weights
        self.conv_layer(self.input)

        #Apply target weights to filters
        self.conv_layer.set_weights([self.weight_arr, self.bias_arr])
        self.conv_layer.trainable = False

    def make_weights(self, kernel_size, num_filters):
        self.weight_arr = np.random.rand(kernel_size, 1, num_filters)
        self.bias_arr = np.zeros(num_filters)
        

    #Iterate through feature map and store filter indicies of max activation for each time step
    def max_act(self):
        self.num_batches = self.out_shape[0]
        activation_list = [] #List of indices of dominant filter activation for each batch (timestep in this case)
        for i in range(self.num_batches):
            if np.max(self.output[i,:,:]) > 0.2 * self.k_size:
                activation_list.append(np.argmax(self.output[i,:,:]))
            else: 
                activation_list.append(self.num_filters)

        self.activation_list = np.array(activation_list)

        return(self.activation_list)


    #construct adjacency matrix
    def adj_matrix(self):
        self.max_act() #Store activation list as object variable by calling max_act()

        a_matrix = np.zeros((self.num_filters + 1, self.num_filters + 1))
        self.list_len = len(self.activation_list)
        #Fill adjacency matrix with the row specifying the filter that appears first
        for i in range(self.list_len):
            if i < self.list_len - 1:
                a_matrix[self.activation_list[i], self.activation_list[i + 1]] += 1 #Add a count to element of adjaceny matrix for each occurence of a sequence of max filter activtions
        
        self.a_matrix = a_matrix

        return self.a_matrix
    
    #Construct transition matrix
    def trans_matrix(self):
        self.adj_matrix()
        sums_arr = np.sum(self.a_matrix, axis=1)[:,np.newaxis]

        self.transition_matrix = np.divide(self.a_matrix, sums_arr, out=np.zeros_like(self.a_matrix), where=sums_arr!=0)

        return self.transition_matrix
    
    def filter_plot3d(self):
            xdim = np.shape(self.weight_arr)[0]  # k_size
            ydim = np.shape(self.weight_arr)[2]  # num filters
            rep_counts = np.full(xdim, ydim)

            xdata = np.repeat(np.arange(1,self.k_size + 1, 1), rep_counts)
            ydata = np.tile(np.arange(1, self.num_filters + 1), self.k_size)
            zdata = self.weight_arr.flatten()

            #Configure plot.
            self.filter_fig = go.Figure()
            self.filter_fig.update_layout(
            scene=dict(
                xaxis=dict(title="Weight"), 
                yaxis=dict(title="Filter"),
                zaxis=dict(title="Magnitude")
            )
        )
            self.filter_fig.add_trace(go.Scatter3d(x=xdata, y=ydata, z=zdata, mode="markers", marker=dict(color='black', symbol="square")))

            self.filter_fig.show()

    def filter_plot2d(self):
        xdata = np.arange(1, self.k_size + 1, 1)

        for filter in range(self.num_filters):
            ydata = self.weight_arr[:,:,filter].flatten()
            
            fig = go.Figure(data=[go.Scatter(x=xdata, y=ydata, mode='markers', marker=dict(color='black', size=40, symbol='square'))])
            fig.update_layout(title=f"Filter {filter + 1} weights",
                              xaxis_title="Weight Number",
                              yaxis_title="Weight Value")
            fig.show()

    def markov_chain(self):
        states = [f"Filter {i+1}" for i in range(self.num_filters + 1)]

        #Check if the transition matrix exists as an object attribute
        if hasattr(self, 'transition_matrix') == False:
            self.transition_matrix = self.trans_matrix()

        #Define and draw markov chain
        mc = MarkovChain(self.transition_matrix, states, self.weight_arr)
        mc.draw()
        



#Example Code:
if __name__ == "__main__":
    time_series = np.random.rand(2000) #Define random time series

    conv_obj = conv_train(time_series, 6) #Define conv_markov object for processing with time series and kernel size of 6

    conv_obj.set_conv(15) #Define convolutional layer with 15 fiters

    feature_map = conv_obj.features() #Get feature map
    weights = conv_obj.top_filters(top_k=5)

    #Set user input weights to filters
    mark_obj = conv_train(time_series, 6) #Establish data object
    mark_obj.my_weights(weights) #Input weights previously obtained from top_filters()
    mark_obj.features() #Get output 
    trans_matrix = mark_obj.trans_matrix() #Retrieve transition matrix

    print(np.sum(trans_matrix)) #Ensure the sum of the transition matrix is 1



#Write function to visualize markov chain with states 

#Edit node class to take a float between 0 and 1 as an argument and set node color based on it

# colored by their dominant pattern: 
# (green: end point higher than average and start, 
# red end point lower than average and start, 
# grey: end point within 1 stdev of average and close to start). 
# Or do gradient from red to green based on 
# how much higher than average the kernel ends at.

#Write function to sample live stock data

#Write function to continuously return the current state that stock is in 






       






