import pandas as pd
import numpy as np
import random
from collections import deque
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load previous model
model_path = "VALID_trading_model_2.keras"
new_model_path = "VALID_trading_model_3.keras"
q_network = load_model(model_path)
target_network = load_model(model_path)

holding_duration = 0

# Set epsilon manually
epsilon = 1.0
num_episodes = 50
epsilon_min = 0.01
epsilon_decay = 0.99995
gamma = 0.99  
batch_size = 32  
memory = deque(maxlen=2000)  
state_size = 5  
action_size = 3  
data_file = "price_data.csv"  

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
        
        latest_row = data.iloc[-1]
        return latest_row['price'], latest_row['volume'], latest_row['market_cap']
    
    def reset(self):
        self.balance = 1000
        self.holdings = 0
        self.holding_duration = 0
        self.current_step = 0
        self.current_price, _, _ = self.read_latest_data()
        return self.get_state()

    def step(self, action):
        self.current_price, volume, market_cap = self.read_latest_data()
        reward = 0
        reward_Subtract = 0
        reward_Bonus = 0
        done = False

        if action == 1:
            if self.balance > 0:

                

                self.holdings += self.balance / self.current_price
                transaction_cost = self.holdings * 0.001
                self.balance = 0
                self.buy_price = self.current_price
                self.holding_duration = 0
                print(f"Buy executed: New Holdings: {self.holdings}, New Balance: {self.balance}")
            
            elif self.holdings > 0:
                print("ERROR BOUGHT WHEN ALREADY HOLDING")
                reward_Subtract += 5

        elif action == 2:
            if self.holdings > 0:
                self.balance += self.holdings * self.current_price

                if self.current_price > self.buy_price:
                    reward_Bonus += (self.current_price - self.buy_price) * 0.4  # Bonus reward for selling at a profit
                    print("BONUS REWARD POSITIVE SELL")

                else:
                    reward_Bonus += (self.current_price - self.buy_price) * 0.4
                    print("NEGATIVE REWARD NEGATIVE / NO VALUE SELL")

                self.holdings = 0
                print(f"Sell executed: New Balance: {self.balance}, Holdings cleared.")

            else:
                reward_Subtract += 5  
                print("ERROR SOLD WHEN NOT HOLDING")

        if self.balance == self.initial_value:
            self.holding_duration += 1
            
            if self.holding_duration >= 35:
                reward_Subtract += 5
                print("HELD INITIAL TOO LONG")
            
            print("INITIAL VALUE")

        portfolio_value = self.balance + (self.holdings * self.current_price)
        reward = portfolio_value - 1000 - reward_Subtract + reward_Bonus - transaction_cost

        next_state = np.array([self.current_price, volume, market_cap, self.holdings, self.balance])

        self.current_step += 1
        if self.current_step >= 200:
            done = True

        return next_state, reward, done
    
    def get_state(self):
        self.current_price, volume, market_cap = self.read_latest_data()
        return np.array([self.current_price, volume, market_cap, self.holdings, self.balance])

def should_stop():
    return os.path.exists("stop.txt")

env = CryptoTradingEnv(data_file)

portfolio_values = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(200):
        if should_stop():
            print("Training stopped by user.")
            break

        if np.random.rand() < epsilon:
            action = random.choice([0, 1, 2])
        else:
            action = np.argmax(q_network.predict(state.reshape(1, -1))[0])

        next_state, reward, done = env.step(action)
        portfolio_value = env.balance + (env.holdings * env.current_price)
        portfolio_values.append(portfolio_value)

        # Print step log
        action_str = ["Hold", "Buy", "Sell"][action]
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

    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    if episode % 2 == 0:
        target_network.set_weights(q_network.get_weights())

q_network.save(new_model_path)
print(f"Model saved as {new_model_path}")

plt.plot(portfolio_values)
plt.title('Portfolio Value Over Time')
plt.xlabel('Step')
plt.ylabel('Portfolio Value')
plt.show()

if not should_stop():
    print("Training completed!")
