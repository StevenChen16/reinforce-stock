# Stock Trading AI

This project implements a Deep Q-Network (DQN) based AI for stock trading simulation using Chinese stock market data.

## Data

The project uses Chinese stock market data, which needs to be downloaded from either of these sources:
- [Kaggle Dataset](https://www.kaggle.com/datasets/stevenchen116/stockchina)
- [Hugging Face Dataset](https://huggingface.co/datasets/StevenChen16/Stock-China-daily)

After downloading, you should have two directories:
- `original_data`: Contains the raw stock data
- `processed_data`: Contains pre-processed stock data

Place these directories in the root of the project.

Note: The `data` directory in this project contains sample data for demonstration purposes only.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/StevenChen16/reinforce-stock.git
   cd reinforce-stock
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the data:
   - Download the dataset from one of the sources mentioned above.
   - Extract the downloaded files.
   - Place the `original_data` and `processed_data` directories in the root of this project.

4. (Optional) If you want to re-process the data:
   ```
   python preprocess.py
   ```
   This step is optional as the downloaded `processed_data` is ready to use.

## Usage

To train the model:

```
python train.py
```

You can modify the training parameters in `train.py` as needed.

## Project Structure

- `train.py`: Main script for training the DQN model
- `environment.py`: Contains the stock trading environment and DQN model definition
- `utils.py`: Utility functions and classes, including the Prioritized Experience Replay buffer
- `preprocess.py`: Script for processing raw stock data (optional use)
- `requirements.txt`: List of Python package dependencies
- `data/`: Directory containing sample data (for demonstration only)
- `processed_data/`: Directory for pre-processed stock data (to be downloaded)
- `original_data/`: Directory for raw stock data (to be downloaded)

## License

Copyright 2023 [Steven Chen]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.