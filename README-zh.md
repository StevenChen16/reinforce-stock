# 股票交易 AI

这个项目实现了一个基于深度 Q 网络（DQN）的 AI，用于使用中国股票市场数据进行股票交易模拟。

## 数据

本项目使用中国股票市场数据，需要从以下任一来源下载：
- [Kaggle 数据集](https://www.kaggle.com/datasets/stevenchen116/stockchina)
- [Hugging Face 数据集](https://huggingface.co/datasets/StevenChen16/Stock-China-daily)

下载后，你应该有两个目录：
- `original_data`：包含原始股票数据
- `processed_data`：包含预处理后的股票数据

请将这些目录放在项目的根目录下。

注意：项目中的 `data` 目录仅包含用于演示目的的样本数据。

## 设置

1. 克隆此仓库：
   ```
   git clone https://github.com/StevenChen16/reinforce-stock.git
   cd reinforce-stock
   ```

2. 安装所需的依赖项：
   ```
   pip install -r requirements.txt
   ```

3. 下载数据：
   - 从上述提到的任一来源下载数据集。
   - 解压下载的文件。
   - 将 `original_data` 和 `processed_data` 目录放在此项目的根目录下。

4. （可选）如果你想重新处理数据：
   ```
   python preprocess.py
   ```
   这一步是可选的，因为下载的 `processed_data` 已经可以直接使用。

## 使用方法

要训练模型，运行：

```
python train.py
```

你可以根据需要修改 `train.py` 中的训练参数。

## 项目结构

- `train.py`：训练 DQN 模型的主脚本
- `environment.py`：包含股票交易环境和 DQN 模型定义
- `utils.py`：实用函数和类，包括优先经验回放缓冲区
- `preprocess.py`：处理原始股票数据的脚本（可选使用）
- `requirements.txt`：Python 包依赖列表
- `data/`：包含样本数据的目录（仅用于演示）
- `processed_data/`：预处理股票数据的目录（需要下载）
- `original_data/`：原始股票数据的目录（需要下载）

## 许可证

版权所有 2023 [Steven Chen]

根据 Apache 许可证 2.0 版（"许可证"）获得许可；
除非遵守许可证，否则您不得使用此文件。
您可以在以下位置获得许可证副本：

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"按原样"分发的，
不附带任何明示或暗示的保证或条件。
有关许可证下的特定语言管理权限和限制，请参阅许可证。