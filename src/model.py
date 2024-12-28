import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


'''
    num_layers: number of hidden layers in the network
    width: number of neurons in each hidden layer
    batch_size: number of samples used in each training batch
    learning_rate: speed at which the model weights are updates during training
'''
class TrainModel:
    '''
        initailize the model with:
        input_dim (size of input vector),
        output_dim (size of the output vector (number of actions))
        batch_size
        learning_rate
    '''
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)


    def _build_model(self, num_layers, width):
        '''
        Build and compile a fully connected deep neural network
        '''
        # keras.Input specifies the input shape of the network
        inputs = keras.Input(shape=(self._input_dim,))
        # adds Dense layers using ReLU activation 
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return model
    

    def predict_one(self, state):
        '''
        Predict the action values from a single state
        '''
        # reshape the state from a 1D array to a 2D array based on required input shape
        state = np.reshape(state, [1, self._input_dim])
        # returns the predicted Q-values for the state
        return self._model.predict(state)


    def predict_batch(self, states):
        '''
        Predict the action values from a batch of states
        '''
        # similar to predict_one but for a batch of states
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        '''
            Train the nn using the updated q-values
        '''
        '''
            fits the model on the batches of data:
                states: input data (features)
                q_sa: target q values
            uses one epoch per call. the network processes the entire batch once
        '''
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        '''
            Save the current model in the folder as h5 file and a model architecture summary as png
        '''
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


'''
    TestModel class is used to load a trained model from a folder and make predictions.
    reuses a previously trained model to make predictions on new data
    ensures the same architecture and weights are used
'''
class TestModel:
    '''
        initailizes the input dimensions and calls _load_my_model() to load
        the saved model from the model_path
    '''
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        '''
        Load the model stored in the folder specified by the model number, if it exists
        '''
        '''
            checks if the model file exists in the folder
            if it exists, loads the model and returns it
            else, exits the program
        '''
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        '''
        Predict the action values from a single state, reshaping the state to match the input shape
        '''
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim