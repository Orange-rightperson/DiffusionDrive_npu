#```python
from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
import ray
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache

logger = logging.getLogger(__name__)

# Import and configure NPU
import torch_npu
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
def initialize_npu():
    # Only set up once per process
    torch_npu.set_device(0)
    torch_npu.empty_cache()

# Cache a single agent instance per process to avoid repeated NPU init
_global_agent = None

def get_global_agent(cfg):
    global _global_agent
    if _global_agent is None:
        initialize_npu()
        _global_agent = instantiate(cfg.agent)
        # call original initialize() if exists
        try:
            _global_agent.initialize()
        except Exception:
            pass
    return _global_agent

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, \
        "Simulator and scorer proposal sampling must match"

    # Use singleton agent with NPU init
    agent = get_global_agent(cfg)

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens_to_evaluate):
        logger.info(f"Processing scenario {idx+1}/{len(tokens_to_evaluate)}")
        score_row = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception:
            logger.warning(f"Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False
        pdm_results.append(score_row)

    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    ray.init(num_cpus=16, ignore_reinit_error=True)
    build_logger(cfg)
    worker = instantiate(cfg.worker)

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    missing = set(scene_loader.tokens) - set(metric_cache_loader.tokens)
    unused = set(metric_cache_loader.tokens) - set(scene_loader.tokens)
    if missing:
        logger.warning(f"Missing metric cache for {len(missing)} tokens.")
    if unused:
        logger.warning(f"Unused metric cache for {len(unused)} tokens.")

    data_points = [
        {"cfg": cfg, "log_file": log_file, "tokens": tokens_list}
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    logger.info(f"Starting pdm scoring of {len(data_points)} logs...")

    score_rows = worker_map(worker, run_pdm_score, data_points)

    df = pd.DataFrame(score_rows)
    success = df["valid"].sum()
    total = len(df)
    avg = df[df["valid"]].drop(columns=["token","valid"]).mean()
    avg_row = avg.to_dict()
    avg_row.update({"token":"average","valid":True})
    df.loc[len(df)] = avg_row

    save_dir = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    df.to_csv(save_dir / f"{timestamp}.csv")
    logger.info(f"Results saved to {save_dir}/{timestamp}.csv")

if __name__ == "__main__":
    main()
#```

