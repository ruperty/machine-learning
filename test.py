# plot a Sequential keras model with a Dense layer of 20 and a Dense layer of 1

# import the required libraries
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model


# create a keras model with a single input and output
model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(1))

# compile the model
#model.compile(optimizer='adam', loss='mean_squared_error')
# print the structure of the model
model.summary()

# plot an image of the model
plot_model(model, to_file='model.png', show_shapes=True)

