# create a keras model with a single input and output
# the input should be an array with one float value
# the output should be an array with one float value and a linear activation function
# randomize the weights of the model
# print the weights of the model
# print the structure of the model
# compile the model
# run predict with a sample input and print the output


# import the required libraries
import numpy as np
import tensorflow as tf

# create a keras model with a single input and output
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1, activation='linear')
])

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# print the structure of the model
model.summary()

# plot an image of the model
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)


# randomize the weights of the model
# model.set_weights([np.random.rand(1,1), np.random.rand(1)])

model.set_weights([np.array([[0.5]]), np.array([0.0])])
print("Model weights after setting: ", model.get_weights())

# run predict with a sample input and print the output
input = np.array([1.0])
# output = model.predict(input)
output = model(input)
print(f"Predicted output for input {input[0]}: {output[0][0]}")

# print the description of calculation of the input values with the weights and the activation function
print("input: ", input)
print("weights: ", model.get_weights())
print("output: ", output)
print("output = input * weights[0] + weights[1] = ", input * model.get_weights()[0] + model.get_weights()[1])


