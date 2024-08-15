import gym
import pygame
import time

def visualize_agent(agent, bins, episodes=5):
    env = gym.make('CartPole-v1', render_mode='human')
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, bins)
        total_reward = 0
        done = False
        
        while not done:
            env.render()
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, bins)
            state = next_state
            total_reward += reward
            time.sleep(0.05)  # Add a small delay to slow down the visualization
        
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    
    env.close()

# Assuming you have your best_agent and bins defined from your training
visualize_agent(best_agent, bins)