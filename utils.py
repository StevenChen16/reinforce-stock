import logging
from collections import deque
import random
import numpy as np
import torch

# 设置日志记录
logging.basicConfig(filename='error.log', level=logging.WARNING, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def console_print(*args, **kwargs):
    print(*args, **kwargs)

def log_print(*args, **kwargs):
    message = ' '.join(map(str, args))
    logging.warning(message)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

def train(env, model, target_model, replay_buffer, optimizer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return 0

    samples, indices, weights = replay_buffer.sample(batch_size)
    state, action, reward, next_state, done = zip(*samples)

    state = torch.FloatTensor(np.array(state)).to(device)
    next_state = torch.FloatTensor(np.array(next_state)).to(device)
    action = torch.LongTensor(np.array(action)).to(device)
    reward = torch.FloatTensor(np.array(reward)).to(device)
    done = torch.FloatTensor(np.array(done)).to(device)
    weights = torch.FloatTensor(weights).to(device)

    q_values = model(state)
    next_q_values = model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(2, action.unsqueeze(2)).squeeze(2)
    next_q_value = next_q_state_values.max(2)[0]

    expected_q_value = reward.unsqueeze(1) + gamma * next_q_value * (1 - done.unsqueeze(1))
    loss = (q_value - expected_q_value.detach()).pow(2).mean(1) * weights
    priorities = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    replay_buffer.update_priorities(indices, priorities.detach().cpu().numpy())

    return loss.item()