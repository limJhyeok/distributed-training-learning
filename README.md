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
