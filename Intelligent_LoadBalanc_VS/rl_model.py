import os
import numpy as np
import tensorflow as tf
import gym
from gym import spaces
from collections import deque

# Uncomment and configure GPU settings if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and memory growth is set.")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("GPU is not available, using CPU.")

# Ensure the 'model/' directory exists
if not os.path.exists('model'):
    os.makedirs('model')

class LoadBalancerEnv(gym.Env):
    def __init__(self):
        super(LoadBalancerEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.float32)

    def reset(self):
        self.state = np.random.rand(3, 3)
        return self.state

    def step(self, action):
        reward = self.calculate_reward(action)
        self.state = np.random.rand(3, 3)
        done = False
        return self.state, reward, done, {}

    def calculate_reward(self, action):
        client_state = self.state[action]
        reward = -client_state[2]
        return reward

def create_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

def train_model(env, model, episodes=1000):
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    memory = deque(maxlen=2000)  # Use deque for efficient memory handling
    episode_rewards = []

    for episode in range(episodes):
        print(f"Starting Episode: {episode}")
        state = env.reset().flatten()
        total_reward = 0

        for step in range(200):
            #print(f"Starting Step: {step}")
            if np.random.rand() <= epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                q_values = model.predict(np.array([state]), verbose=0)
                action = np.argmax(q_values[0])

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                # Ensure memory elements have consistent shapes
                minibatch = np.random.choice(len(memory), batch_size, replace=False)
                
                # Convert the elements to numpy arrays
                states = np.array([memory[i][0] for i in minibatch])
                actions = np.array([memory[i][1] for i in minibatch])
                rewards = np.array([memory[i][2] for i in minibatch])
                next_states = np.array([memory[i][3] for i in minibatch])
                dones = np.array([memory[i][4] for i in minibatch]).astype(int)  # Convert booleans to integers

                # Calculate target values
                targets = rewards + gamma * np.amax(model.predict(next_states, verbose=0), axis=1) * (1 - dones)
                targets_full = model.predict(states, verbose=0)

                # Update the Q-values for the taken actions
                for i, action in enumerate(actions):
                    targets_full[i][action] = targets[i]
                
                # Train the model
                model.fit(states, targets_full, verbose=0)

            if done:
                break

        # Decay the exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    print(f"Training Complete. Average Reward per Episode: {np.mean(episode_rewards):.2f}")
    model.save("model/load_balancer_MAINmodel.keras")

if __name__ == "__main__":
    env = LoadBalancerEnv()
    model = create_model(input_shape=env.observation_space.shape[0] * env.observation_space.shape[1], output_shape=env.action_space.n)
    train_model(env, model)
