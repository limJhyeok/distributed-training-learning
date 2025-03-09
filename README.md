# DDP Learning


Repository for learning DDP(Data Distributed Parallel)

## Usage
### Install package
```bash
pip install uv
uv sync
```

### message passing
```bash
torchrun --nnodes=1 --nproc-per-node=2 src/msg_pass.py
```

### ddp train demo

**performs DDP forward & backward using simple Linear model** and logs key information such as **inputs, outputs, loss, gradients, and updated weights** in a JSON file for each rank.

#### **Key Features**
- Init random tensors(inputs & labels) and (forward & backward) a model using **DDP**
- Logs **input data, loss values, gradients, and updated weights** for each rank
- Saves results in a JSON file (`logs/ddp_rank-<rank>.json`)

#### **Usage**
```bash
torchrun --nnodes=1 --nproc-per-node=2 src/ddp_train.py
```
**Example Output File:** (`logs/ddp_rank-0.json`, `logs/ddp_rank-1.json`)
