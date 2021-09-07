## 实验框架使用说明

### 1. 使用环境

所有 Python 依赖包，均在根目录下 `requirements.txt` 中有，通过：

```bash
conda install --yes --file requirements.txt
```

或者（建立虚拟环境）：

```bash
conda create --name <env> --file requirements.txt
```

即可完成环境配置。

实验使用的 Python 版本为 Python 3.8.8。

或可通过：

```bash
pip3 install -r requirements_pip.txt
```

也可安装所有环境依赖包。

### 2. 基本使用

在根目录下运行：

```
python3 main.py --model <model-name>
```

即可按照默认参数运行对应名字的模型。

### 3. 模型名字列表

上述 `<model-name>` 包括：

- `ConvModel`
- `RNNModel`
- `RNNAttentionModel`
- `MLPModel`
- `SelfAttentionModel`
- `SelfAttentionWithPosModel`

### 4. 其他可选参数

所有模型可选参数包括：

- `--lr <rate>`：设置学习率，默认 $10^{-3}$。
- `--optimizer_name <name>`：设置优化器，默认 `Adam`，可选 `GD, Adam, Adagrad` 等。
- `--l2 <rate>`：设置 L2 正则化系数，默认 $10^{-6}$。
- `--batch_size <num>`：设置 Batch Size，默认 $128$。
- `--max_epoch <num>`：设置最大训练 Epoch 数，默认 $200$。
- `--es_patience <num>`：设置 Early Stopping 的 Epoch 数阈值，默认 $10$。
- `--feature_num <num>`：设置词向量维度，默认 $64$。
- `--dropout_rate <rate>`：设置 Dropout 层的概率，默认 $0.5$。

对于 CNN（`ConvModel`）：

- `--out_channels <num>`：设置每个 `filter` 输出的 Channels，默认 $100$。
- `--filters <list>`：设置每个 `filter` 的大小，默认 `[3, 4, 5]`。

对于 RNN（`RNNModel`）：

- `--num_layers <num>`：设置 RNN 中隐藏层数，默认 $2$。
- `--hidden_size <num>`：设置 RNN 中隐藏层维度大小，默认 $64$。
- `--RNN_type <name>`：设置 RNN 类型，默认 `GRU`，可选 `RNN, LSTM, GRU` 等。

对于 RNN-Attention（`RNNAttentionModel`），在 RNN 基础上增加：

- `--att_size <num>`：设置 Attention 过程隐藏层维度大小，默认 $64$。

对于 MLP（`MLPModel`）：

- `--hidden_size <num>`：设置 MLP 隐藏层大小， 默认 $64$。

对于 SelfAttention（`SelfAttentionModel`），在 RNN-Attention 除去 `--RNN_type` 基础上增加：

- `--num_heads <num>`：设置 MultiHeadAttention 中的 Head 数量，默认 $1$。

特别地，可以使用 `--test Yes`（严格大小写）来使得模型只根据本地保存的 Checkpoint 进行测试集上性能测试，而不进行训练。

### 5. 常用方法举例

```bash
python3 main.py --model RNNAttention --RNN_type GRU
python3 main.py --model ConvModel --lr 0.01 --l2 0
python3 main.py --model MLPModel --feature_num 128 --test Yes
```

