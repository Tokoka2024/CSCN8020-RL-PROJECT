import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym

# Load the saved model with custom objects
model = keras.models.load_model('ddqn_cartpole_model2.h5', custom_objects={'mse': keras.losses.MeanSquaredError()})

# Function to visualize the agent in the environment
def visualize_saved_model(model, env, n_episodes=5):
    for episode in range(n_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        total_reward = 0

        for time in range(500):
            env.render()
            action = np.argmax(model.predict(state, verbose=0))
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
                break
    env.close()

# Initialize the environment with render_mode
env = gym.make('CartPole-v1', render_mode='human')

# Visualize the saved model
visualize_saved_model(model, env)
