import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from utils import console_print, log_print

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, num_stocks):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, output_dim * num_stocks)
        self.num_stocks = num_stocks
        self.output_dim = output_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x.view(-1, self.num_stocks, self.output_dim)

class StockSimulator:
    def __init__(self, data_dir):
        self.data = self.load_data(data_dir)
        if not self.data:
            raise ValueError("No data was loaded. Check your data directory and file formats.")
        self.all_dates = sorted(set(self.data[next(iter(self.data))].index))
        self.normalize_data()
        self.reset()

    def load_data(self, data_dir):
        data = {}
        log_print(f"Attempting to load data from directory: {data_dir}")
        files = os.listdir(data_dir)
        log_print(f"Files found in directory: {files}")

        for file in files:
            if file.endswith('.csv'):
                stock_name = file.split('_')[0]
                file_path = os.path.join(data_dir, file)
                log_print(f"Attempting to load file: {file_path}")

                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    df = df.dropna()
                    df = df.replace([np.inf, -np.inf], np.nan).dropna()

                    if df.empty:
                        log_print(f"Warning: {file} has no valid data after cleaning.")
                        continue

                    data[stock_name] = df
                    log_print(f"Successfully loaded data for stock: {stock_name}")
                except Exception as e:
                    log_print(f"Error loading file {file}: {str(e)}")

        if not data:
            log_print("No valid data was loaded. Check your data directory and file formats.")
        else:
            log_print(f"Successfully loaded data for {len(data)} stocks.")

        return data

    def normalize_data(self):
        for stock in self.data:
            for column in ['open', 'high', 'low', 'close', 'volume']:
                mean = self.data[stock][column].mean()
                std = self.data[stock][column].std()
                if std == 0:
                    log_print(f"Warning: Zero standard deviation for {stock} {column}")
                    self.data[stock][column] = 0
                else:
                    self.data[stock][column] = (self.data[stock][column] - mean) / std

    def reset(self):
        self.portfolio = {}
        self.cash = 1000000
        self.transaction_history = []
        self.current_date_index = 0
        self.done = False
        self.current_step = 0
        self.start_date_index = None
        self.end_date_index = None
        self.previous_total_assets = self.cash
        self.daily_records = []

    def find_closest_date_index(self, target_date):
        closest_date = min(self.all_dates, key=lambda x: abs(x - target_date))
        return self.all_dates.index(closest_date)

    def set_dates(self, start_date, end_date):
        self.start_date_index = self.find_closest_date_index(start_date)
        self.end_date_index = self.find_closest_date_index(end_date)
        self.current_date_index = self.start_date_index
        self.previous_total_assets = self.calculate_total_assets(self.current_date)
        print(f"Setting dates: start_date_index={self.start_date_index}, end_date_index={self.end_date_index}")

    @property
    def current_date(self):
        if self.current_date_index >= len(self.all_dates) or self.current_date_index < 0:
            print(f"Invalid current_date_index: {self.current_date_index}, all_dates length: {len(self.all_dates)}")
            return None
        return self.all_dates[self.current_date_index]

    def get_state(self, stock):
        if self.current_date is None:
            return [0] * 5
        state = []
        df = self.data[stock]
        
        current_day = df[df.index == self.current_date]
        if not current_day.empty:
            state.extend([
                current_day.iloc[0]['close'],
                current_day.iloc[0]['MA5'],
                current_day.iloc[0]['RSI'],
                self.portfolio.get(stock, {'shares': 0})['shares'],
                self.cash / self.previous_total_assets
            ])
        else:
            state = [0] * 5
        
        return state

    def step(self, actions):
        if self.current_date is None:
            print("Step called with invalid current_date")
            return [], 0, True, {}
        
        daily_trades = []
        daily_pnl = 0
        for stock, action in actions.items():
            amount = action - 1
            if self.trade_stock(self.current_date, stock, amount):
                daily_trades.append((stock, amount))

        self.current_step += 1
        self.current_date_index += 1
        if self.current_date_index > self.end_date_index:
            self.done = True

        next_state = [self.get_state(stock) for stock in self.data.keys()]
        next_state = [item for sublist in next_state for item in sublist]
        total_assets = self.calculate_total_assets(self.current_date)
        daily_pnl = total_assets - self.previous_total_assets
        self.previous_total_assets = total_assets

        self.daily_records.append({
            'date': self.current_date,
            'trades': daily_trades,
            'portfolio': self.portfolio.copy(),
            'total_assets': total_assets,
            'cash': self.cash,
            'daily_pnl': daily_pnl
        })

        reward = self.calculate_reward(daily_pnl, daily_trades)
        
        return next_state, reward, self.done, {'total_assets': total_assets}

    def trade_stock(self, date, stock, amount):
        if stock not in self.data:
            log_print(f"Stock {stock} not found.")
            return False

        stock_data = self.data[stock][self.data[stock].index == date]
        if stock_data.empty:
            return False

        close_price = stock_data['close'].values[0]
        total_cost = close_price * amount

        if amount > 0:
            if total_cost > self.cash:
                return False
            if stock not in self.portfolio:
                self.portfolio[stock] = {'shares': 0, 'avg_cost': 0}
            self.cash -= total_cost
            self.portfolio[stock]['shares'] += amount
            self.portfolio[stock]['avg_cost'] = (self.portfolio[stock]['avg_cost'] * (self.portfolio[stock]['shares'] - amount) + total_cost) / self.portfolio[stock]['shares']
            log_print(f"Bought {amount} shares of {stock} at {close_price}")
            return True
        elif amount < 0:
            amount = abs(amount)
            if stock not in self.portfolio or self.portfolio[stock]['shares'] < amount:
                return False
            self.portfolio[stock]['shares'] -= amount
            if self.portfolio[stock]['shares'] == 0:
                del self.portfolio[stock]
            self.cash += close_price * amount
            log_print(f"Sold {amount} shares of {stock} at {close_price}")
            return True
        return False

    def calculate_reward(self, daily_pnl, daily_trades):
        reward = 0
        
        base_reward = daily_pnl / self.previous_total_assets
        reward += base_reward * 100

        if daily_pnl > 0:
            if daily_pnl / self.previous_total_assets >= 0.01:
                reward += 100
            elif daily_pnl / self.previous_total_assets >= 0.005:
                reward += 50

        if daily_pnl < 0:
            if abs(daily_pnl) / self.previous_total_assets > 0.01:
                reward -= 50
            else:
                reward -= 20

        trade_count = len(daily_trades)
        small_trades = 0
        for _, amount in daily_trades:
            if abs(amount) == 1:
                small_trades += 1
            elif abs(amount) > 1:
                reward += min(abs(amount), 10) * 2

        reward -= small_trades * 10

        if sum([info['shares'] for info in self.portfolio.values()]) == 0:
            reward -= 50

        cash_ratio = self.cash / self.previous_total_assets
        if cash_ratio > 0.5:
            reward -= 30
        elif cash_ratio < 0.1:
            reward -= 20

        portfolio_size = len(self.portfolio)
        if portfolio_size == 0:
            reward -= 50
        elif portfolio_size == 1:
            reward -= 30
        elif 2 <= portfolio_size <= 5:
            reward += 20
        elif portfolio_size > 5:
            reward += 50

        if trade_count > 30:
            reward -= (trade_count - 30) * 2

        return reward

    def calculate_total_assets(self, date):
        total_value = self.cash
        for stock, info in self.portfolio.items():
            stock_data = self.data[stock][self.data[stock].index == date]
            if not stock_data.empty:
                close_price = stock_data['close'].values[0]
                total_value += close_price * info['shares']
        return total_value