import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm

from environment import StockSimulator, DQN
from utils import PrioritizedReplayBuffer, train, console_print

def main(continue_training=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on {}".format(device))

    env = StockSimulator(data_dir='data_more_processed')
    env.set_dates(start_date=pd.Timestamp('2022-01-01'), end_date=pd.Timestamp('2023-01-01'))

    single_state = env.get_state(list(env.data.keys())[0])
    console_print(f"Single state shape: {np.array(single_state).shape}")

    input_dim = len(single_state) * len(env.data)
    output_dim = 3  # 买入、持有、卖出
    num_stocks = len(env.data)

    model = DQN(input_dim, output_dim, num_stocks).to(device)
    target_model = DQN(input_dim, output_dim, num_stocks).to(device)

    if continue_training:
        if os.path.exists('best_dqn_model_512.pth'):
            model.load_state_dict(torch.load('best_dqn_model_512.pth'))
            target_model.load_state_dict(torch.load('best_dqn_model.pth'))
            print("Loaded previous model weights.")
        else:
            print("No previous model weights found. Training from scratch.")
    else:
        print("Training from scratch.")

    target_model.load_state_dict(model.state_dict())

    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    replay_buffer = PrioritizedReplayBuffer(500000)

    batch_size = 1024
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 5000
    target_update = 50
    num_episodes = 2000
    patience = 50
    best_reward = float('-inf')
    episodes_without_improvement = 0

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        env.reset()
        env.set_dates(start_date=pd.Timestamp('2020-01-01'), end_date=pd.Timestamp('2023-01-01'))
        total_reward = 0
        losses = []

        epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * episode / epsilon_decay)

        for step in tqdm(range(env.start_date_index, env.end_date_index), desc=f"Episode {episode}", leave=False):
            states = []
            for stock in env.data.keys():
                state = env.get_state(stock)
                if len(state) == 0:
                    print(f"Empty state encountered at episode {episode}, step {step}, date {env.current_date}")
                    state = [0] * 5  # 使用零填充
                states.extend(state)
            
            if len(states) != input_dim:
                print(f"Incorrect state length: {len(states)}, expected: {input_dim}")
                states = states + [0] * (input_dim - len(states))  # 使用零填充到正确的长度
            
            state_tensor = torch.FloatTensor(states).unsqueeze(0).to(device)
            
            if random.random() > epsilon:
                with torch.no_grad():
                    actions = model(state_tensor).squeeze(0).argmax(1).tolist()
            else:
                actions = [random.randint(0, 2) for _ in range(num_stocks)]

            next_state, reward, done, info = env.step(dict(zip(env.data.keys(), actions)))
            total_reward += reward

            replay_buffer.push(states, actions, reward, next_state, done)

            loss = train(env, model, target_model, replay_buffer, optimizer, batch_size, gamma, device)
            if loss is not None and not np.isnan(loss):
                losses.append(loss)

            if step % 100 == 0:
                console_print(f"Episode {episode}, Step {step}, Total Reward: {total_reward:.2f}, Loss: {loss:.4f}")

            if env.done:
                break

        avg_loss = np.mean(losses) if losses else 0
        console_print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Average Loss: {avg_loss:.4f}")

        if not np.isnan(total_reward) and total_reward > best_reward:
            best_reward = total_reward
            episodes_without_improvement = 0
            torch.save(model.state_dict(), 'best_dqn_model.pth')
        else:
            episodes_without_improvement += 1
        
        if episodes_without_improvement >= patience:
            console_print(f"Early stopping at episode {episode}")
            break
        
        scheduler.step()

    # 打印每天的交易和资产情况
    for record in env.daily_records:
        console_print(f"日期: {record['date']}")
        console_print(f"交易: {record['trades']}")
        console_print(f"持仓: {record['portfolio']}")
        console_print(f"总资产: {record['total_assets']:.2f}")
        console_print(f"现金: {record['cash']:.2f}")
        console_print(f"每日盈亏: {record['daily_pnl']:.2f}")
        console_print("-" * 40)

if __name__ == "__main__":
    continue_training = True  # 设置为 True 表示继续训练，设置为 False 表示从头开始训练
    main(continue_training)