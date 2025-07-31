#```python
# navsim/planning/script/run_pdm_score.py
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.agents.base import AbstractAgent
from nuplan.planning.utils.multithreading.worker_utils import worker_map
import torch
# If using PyTorch NPU, ensure device context is set early
import torch_npu

def initialize_npu():
    # Set NPU device 0 (or use cfg to customize)
    torch_npu.set_device(0)
    torch_npu.empty_cache()

# Cache a single agent instance per process
_global_agent: AbstractAgent = None

def get_global_agent(cfg) -> AbstractAgent:
    global _global_agent
    if _global_agent is None:
        # Initialize NPU before agent instantiation
        initialize_npu()
        # Instantiate only once (loads NPU context)
        _global_agent = instantiate(cfg.agent)
    return _global_agent

@hydra.main(config_path="../../planning/config", config_name="run_pdm_score")
def main(cfg):
    # Log effective config
    print(OmegaConf.to_yaml(cfg))

    # Load data points via Hydra loader
    data_points = load_data_points(cfg)

    # Define per-item scoring function
    def score_fn(data_point):
        agent = get_global_agent(cfg)
        return agent.score(data_point)

    # Use worker_map (Ray or serial) but agent instantiates once per process
    score_rows = worker_map(cfg.worker, score_fn, data_points)

    # Save or log results
    save_scores(score_rows, cfg)

if __name__ == "__main__":
    main()
#```

