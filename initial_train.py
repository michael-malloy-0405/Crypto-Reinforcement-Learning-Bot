import pandas as pd
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
import time
import os  # To check for the stop.txt file
import matplotlib.pyplot as plt

stop_training = False
holding_duration = 0

class CryptoTradingEnv:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.balance = 1000
        self.initial_value = self.balance
        self.holdings = 0
        self.current_price = None
        self.current_step = 0

    def read_latest_data(self):
        data = pd.read_csv(self.csv_file_path)
        if data.empty or len(data) == 0:
            raise ValueError("CSV file is empty or not updating")
        
        latest_row = data.iloc[-1]  # Get last row
        print(f"Latest Data: {latest_row}")
        return latest_row['price'], latest_row['volume'], latest_row['market_cap']
    
    def reset(self):
        self.balance = 1000
        self.holdings = 0
        self.current_step = 0
        self.holding_duration = 0
        self.current_price, _, _ = self.read_latest_data()
        return self.get_state()

    def step(self, action):
        """
        Perform the given action (0 = Hold, 1 = Buy, 2 = Sell).
        Returns: next_state, reward, done
        """
        
        self.current_price, volume, market_cap = self.read_latest_data()

        reward = 0
        reward_Subtract = 0
        reward_Bonus = 0
        done = False

        if action == 1:
            if self.balance > 0:
                self.holdings += self.balance / self.current_price
                self.balance = 0
                self.buy_price = self.current_price  # Record the price at which the asset was bought
                self.holding_duration = 0
                print(f"Buy executed: New Holdings: {self.holdings}, New Balance: {self.balance}")

            elif self.holdings > 0:
                print("ERROR BOUGHT WHEN ALREADY HOLDING")
                reward_Subtract += 5

    # Handle Sell Action
        elif action == 2:
            if self.holdings > 0:
                self.balance += self.holdings * self.current_price
                # Reward for selling at a higher price than bought
                if self.current_price > self.buy_price:
                    reward_Bonus += (self.current_price - self.buy_price) * 0.5  # Bonus reward for selling at a profit
                    print("BONUS REWARD POSITIVE SELL")

                else:
                    reward_Bonus += (self.current_price - self.buy_price) * 0.5
                    print("NEGATIVE REWARD NEGATIVE / NO VALUE SELL")
                self.holdings = 0
                print(f"Sell executed: New Balance: {self.balance}, Holdings cleared. Sold at: {self.current_price}")
            else:
                reward_Subtract += 10  
                print("ERROR SOLD WHEN NOT HOLDING")  # Penalty for trying to sell when no holdings

        if self.balance == self.initial_value:
            self.holding_duration += 1
            
            if self.holding_duration >= 35:
                reward_Subtract += 5
                print("HELD INITIAL TOO LONG")
            
            print("INITIAL VALUE")
            
        portfolio_value = self.balance + (self.holdings * self.current_price)
        reward = portfolio_value - 1000 - reward_Subtract + reward_Bonus

        next_state = np.array([self.current_price, volume, market_cap, self.holdings, self.balance])

        self.current_step += 1
        if self.current_step >= 200:
            done = True

        return next_state, reward, done
    
    def get_state(self):
        """Fetch the initial state."""
        self.current_price, volume, market_cap = self.read_latest_data()
        return np.array([self.current_price, volume, market_cap, self.holdings, self.balance])

def build_q_network(state_size, action_size):
    """
    Build a simple Q-network.
    - state_size: Number of features in the state.
    - action_size: Number of possible actions.
    """

    model = Sequential([
        Input(shape=(state_size,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(action_size, activation='linear')  # Output Q-values for each action
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), loss='mse')
    return model

def should_stop():
    return os.path.exists("stop.txt")  # Check if stop.txt exists

gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.99995  # Decay factor for exploration
batch_size = 32  # Size of the replay buffer batch
memory = deque(maxlen=2000)  # Replay buffer

state_size = 5  # State size (price, volume, market cap, holdings, balance)
action_size = 3  # Number of actions (hold, buy, sell)
data_file = "price_data.csv"  # Path to your CSV file
env = CryptoTradingEnv(data_file)  # Create the environment
q_network = build_q_network(state_size, action_size)  # Create the Q-network
target_network = build_q_network(state_size, action_size)  # Create the target network
target_network.set_weights(q_network.get_weights())  # Sync the target network with the Q-network
portfolio_values = []

for episode in range(50):
    state = env.reset()
    total_reward = 0

    for t in range(200):
        if should_stop():  # Check for stop.txt file to stop training
            print("Training stopped by user.")
            break

        if np.random.rand() < epsilon:
            action = random.choice([0, 1, 2])  # Random action for exploration
        else:
            action = np.argmax(q_network.predict(state.reshape(1, -1))[0])  # Predict action based on Q-values

        # Perform action in the environment
        next_state, reward, done = env.step(action)
        
        # Track portfolio value
        portfolio_value = env.balance + (env.holdings * env.current_price)
        portfolio_values.append(portfolio_value)

        # Print the action taken and the portfolio value
        if action == 1:
            action_str = "Buy"
        elif action == 2:
            action_str = "Sell"
        else:
            action_str = "Hold"
        
        print(f"Episode {episode + 1}, Step {t+1}, Action: {action_str}, Holdings: {env.holdings}, Portfolio Value: {portfolio_value}, Reward: {reward}, Epsilon: {epsilon}")

        memory.append((state, action, reward, next_state, done))

        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state_b, action_b, reward_b, next_state_b, done_b in minibatch:
                target = reward_b
                if not done_b:
                    target = reward_b + gamma * np.max(target_network.predict(next_state_b.reshape(1, -1))[0])
                target_f = q_network.predict(state_b.reshape(1, -1))
                target_f[0][action_b] = target
                q_network.fit(state_b.reshape(1, -1), target_f, epochs=1, verbose=0)

        state = next_state
        total_reward += reward

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if done:
            break

    print(f"Episode {episode+1}/{1000}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    if episode % 2 == 0:
        target_network.set_weights(q_network.get_weights())

q_network.save('VALID_trading_model_one.keras')
print("Model saved successfully!")

# After training, you can visualize the portfolio values over time
plt.plot(portfolio_values)
plt.title('Portfolio Value Over Time')
plt.xlabel('Step')
plt.ylabel('Portfolio Value')
plt.show()

if not should_stop():
    print("Training completed!")
