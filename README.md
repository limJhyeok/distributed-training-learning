# Distributed Training Learning


Repository for learning Distributed & Parallel training using PyTorch

**Supported Methods**
- Message Passing(rank 0 -> rank 1)
- DDP(Distributed Data Parallel)
- TP(Tensor Parallelism)
- PP(Pipeline Parallelism)


## Usage
### Prerequisites
- At least 2 GPUs


### Install package
```bash
pip install uv
uv sync
```
---

### message passing
Implements a simple **point-to-point communication.** using **gloo** backend.

- Rank 0 sends a tensor containing the value 42 to all other ranks.
- Other ranks receive the tensor from rank 0 and print the received value.


```bash
torchrun --nnodes=1 --nproc-per-node=2 src/msg_pass.py
```
---
### ddp train

performs **DDP forward & backward** using simple Linear model and logs key information such as **inputs, outputs, loss, gradients, and updated weights** in a JSON file for each rank.

#### **Key Features**
- Init random tensors(inputs & labels) and (forward & backward) a model using **DDP**
- Logs **input data, loss values, gradients, and updated weights** for each rank
- Saves results in a JSON file (`logs/ddp_rank-<rank>.json`)

#### **Usage**
```bash
torchrun --nnodes=1 --nproc-per-node=2 src/ddp_train.py
```
**Example Output File:** (`logs/ddp_rank-0.json`, `logs/ddp_rank-1.json`)

---
### tp train

performs **Tensor Parallelism(TP)** in Linear layer(Matrix Multiplication)

#### **Usage**
```bash
torchrun --nnodes=1 --nproc-per-node=2 src/tp_train.py
```
