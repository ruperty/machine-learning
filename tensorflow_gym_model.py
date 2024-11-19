# create a tensorflow 2.0 model using input from a gym environment
# and output to the gym environment

import tensorflow as tf
import numpy as np
import gym

# create a gym environment
env = gym.make('CartPole-v1')

# define the input and output sizes
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# create a simple model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_size,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# train the model
# create a training loop
for i in range(1000):
    # get the current state
    state = env.reset()
    state = np.reshape(state, [1, input_size])
    done = False
    total_reward = 0
    
    while not done:
        # get the action
        q_values = model.predict(state)
        action = np.argmax(q_values)
        
        # take the action
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, input_size])
        
        # accumulate the reward
        total_reward += reward
        
        # train the model
        target = q_values
        target[0][action] = reward + 0.99 * np.max(model.predict(next_state))
        model.fit(state, target, verbose=0)
        
        state = next_state
        
        # check if the episode is done
        if done:
            break

# save the model

model.save('cartpole_model.h5')

# load the model
model = tf.keras.models.load_model('cartpole_model.h5')

# test the model
state = env.reset()
state = np.reshape(state, [1, input_size])
done = False

while not done:
    action = np.argmax(model.predict(state))
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, input_size])
    state = next_state
    env.render()

env.close()

# end of file
