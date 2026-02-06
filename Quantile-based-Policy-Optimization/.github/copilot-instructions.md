# AI Agent Instructions for Quantile-based-Policy-Optimization

This repository contains the implementation of Quantile-Based Deep Reinforcement Learning using Two-Timescale Policy Gradient Algorithms. Here's what you need to know to work effectively with this codebase:

## Project Structure

The project is organized into several experiment modules:
- `fair_lottery/`: Comparison between QPO/QPPO and mean-based algorithms
- `beyond_greed/`: Comparison between QPO/QPPO and distributional RL algorithms 
- `inventory_management/`: Multi-echelon supply chain inventory management application
- `toy_example/`: Basic experiment with theoretical assumptions satisfied
- `portfolio_management/`: Financial portfolio management application

Each module follows a consistent structure:
```
module_name/
  ├── train.py or train_*.py      # Training entry points
  ├── agents/                     # Agent implementations
  │   ├── __init__.py
  │   ├── qpo.py                 # QPO algorithm
  │   ├── qppo.py               # QPPO algorithm
  │   └── [other agents].py     # Additional algorithms
  ├── envs/                      # Environment definitions
  │   ├── __init__.py
  │   ├── env_wrappers.py       # Optional environment wrappers
  │   └── [env_name].py         # Main environment implementation
  └── utils/                     # Shared utilities
      ├── __init__.py
      ├── memory.py             # Replay buffer implementations
      └── model.py              # Neural network models
```

## Key Components

### Training Configuration
All training scripts use an `Options` class for configuration. Key parameters:
- `q_alpha`: Quantile level (default varies by experiment)
- `est_interval`: Estimation interval for metrics
- `log_interval`: Logging frequency
- `max_episode`: Maximum training episodes
- `gamma`: Discount factor
- `lr`, `q_lr`: Learning rates for policy and quantile estimators

Example from `toy_example`:
```python
parser.add_argument('--q_alpha', type=float, default=0.25)
parser.add_argument('--est_interval', type=int, default=100)
parser.add_argument('--max_episode', type=int, default=200000)
```

### Algorithm Implementation Pattern
- Algorithms inherit from base agent classes
- Each agent has `train()` method as main entry point
- Quantile-based algorithms (QPO/QPPO) use two-timescale learning
- Training logs are saved under `logs/{env_name}/{algo_name}_{timestamp}_{seed}/`

## Developer Workflows

### Running Experiments
1. Single experiment:
```bash
python train.py  # For toy_example, fair_lottery, beyond_greed
python train_[algo].py  # For inventory_management, portfolio_management
```

2. Multi-process training:
All training scripts support parallel training with different seeds using `multiprocessing.Pool`.

### Environment Setup
Required environment:
- Python 3.9
- PyTorch 1.9.0+cu111
- CUDA support recommended for larger experiments

## Conventions and Patterns

### Code Organization
1. Algorithm implementations are isolated in `agents/` directory
2. Environment definitions are in `envs/` directory
3. Shared utilities in `utils/` directory
4. Each module is self-contained but follows consistent structure

### State Management
1. Random seeds are set consistently for reproducibility:
```python
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
```

2. Training states are logged to TensorBoard:
- Located in `logs/{env_name}/{algo_name}_{timestamp}_{seed}/`
- Key metrics tracked at intervals specified by `log_interval`

## Integration Points
1. Algorithm Integration:
   - Implement new algorithms in `agents/` directory
   - Follow existing agent interface patterns
   - Add algorithm to `__init__.py` for easy importing

2. Environment Integration:
   - Add new environments in `envs/` directory
   - Implement standard gym interface
   - Add wrappers in `env_wrappers.py` if needed

## Common Tasks
1. Adding a new experiment:
   - Create new directory following module structure
   - Implement environment in `envs/`
   - Copy and modify training script
   - Add entry in README.md

2. Modifying algorithms:
   - Relevant files in `agents/` directory
   - Check parameter defaults in training script
   - Update configurations in `Options` class

Remember to maintain consistency with the project's organizational structure and coding patterns when making changes.