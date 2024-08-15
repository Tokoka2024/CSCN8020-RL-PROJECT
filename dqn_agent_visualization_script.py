import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('dqn_cartpole_model2.h5', compile=False)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Create the environment
env = gym.make('CartPole-v1', render_mode='human')

# Set random seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Run episodes with the trained agent
n_episodes = 10
for episode in range(n_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, 4])  # Assuming state_size is 4 for CartPole
    done = False
    total_reward = 0

    while not done:
        env.render()  # This will display the environment
        action = np.argmax(model.predict(state, verbose=0)[0])
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        state = next_state
        total_reward += reward
        done = done or truncated

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()