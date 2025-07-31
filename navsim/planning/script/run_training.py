import os
import torch
import torch_npu  # ensure torch_npu is imported
from torch_npu.npu import amp
from torch_npu.npu.amp import GradScaler
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
from pathlib import Path
import logging
from tqdm import tqdm
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
import torch.distributed as dist

def is_main_process():
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

# Build datasets as before
def build_datasets(cfg: DictConfig, agent: AbstractAgent):
    train_filter = instantiate(cfg.train_test_split.scene_filter)
    train_filter.log_names = train_filter.log_names or cfg.train_logs
    val_filter = instantiate(cfg.train_test_split.scene_filter)
    val_filter.log_names = val_filter.log_names or cfg.val_logs
    data_path = Path(cfg.navsim_log_path)
    blobs = Path(cfg.sensor_blobs_path)
    train_loader_src = SceneLoader(blobs, data_path, train_filter, agent.get_sensor_config())
    val_loader_src = SceneLoader(blobs, data_path, val_filter, agent.get_sensor_config())
    train_ds = Dataset(train_loader_src, agent.get_feature_builders(), agent.get_target_builders(), cfg.cache_path, cfg.force_cache_computation)
    val_ds = Dataset(val_loader_src, agent.get_feature_builders(), agent.get_target_builders(), cfg.cache_path, cfg.force_cache_computation)
    return train_ds, val_ds

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    # set device index for NPU
    local_rank = int(os.environ["LOCAL_RANK"]) # 在shell脚本中循环传入local_rank变量作为指定的device
    device = torch.device('npu', local_rank) # local_rank用于自动获取device号
    torch.distributed.init_process_group(backend="hccl",rank=local_rank) # 将通信方式设置为hccl
    torch_npu.npu.set_device(local_rank)
     # 如样例代码所示，定义一个简单的神经网络
    #device_index = int(os.environ.get('DEVICE_ID', 0))
    #Etorch.npu.set_device(device_index)
    #device = torch.device(f'npu:{device_index}')
    # seed
    torch.manual_seed(cfg.seed)
    logger.info(f"Using device: {device}")

    # Build agent and model
    agent: AbstractAgent = instantiate(cfg.agent)
    if hasattr(agent, 'model'):
        model = agent.model.to(device)
    else:
        model = agent.to(device)

    # Optimizer and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer.params)
    scaler = GradScaler()
    start_epoch = 0
    ckpt_dir = "/data/ckpts/navsim"
    last_ckpt_path = os.path.join(ckpt_dir, "checkpoint_epoch_26.pth")
    if os.path.exists(last_ckpt_path):
        logger.info(f" Loading checkpoint from: {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f" Resumed from epoch {start_epoch}")
    # build dataset
    if cfg.use_cache_without_dataset:
        train_ds = CacheOnlyDataset(cfg.cache_path, agent.get_feature_builders(), agent.get_target_builders(), cfg.train_logs)
        val_ds = CacheOnlyDataset(cfg.cache_path, agent.get_feature_builders(), agent.get_target_builders(), cfg.val_logs)
    else:
        train_ds, val_ds = build_datasets(cfg, agent)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
    train_loader = DataLoader(train_ds, **cfg.dataloader.params, shuffle=False, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_ds, **cfg.dataloader.params, shuffle=False, drop_last=True, sampler=val_sampler)
    #idef move_tensor(x):
    #    return x.to(device).float() if isinstance(x, torch.Tensor) else x
    def move_to_device(data):
        return {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    # training loop
    for epoch in range(start_epoch, cfg.trainer.params.max_epochs):
        model.train()
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for features, targets in train_iter:
            features = move_to_device(features)
            targets = move_to_device(targets)
            #def move_tensor(x):
            #    return x.to(device).float() if isinstance(x, torch.Tensor) else x
            #features = batch[0]  # assuming batch = [features, targets]
            #targets = batch[1]
            #features = {k: move_tensor(v) for k, v in features.items()}
            #targets = {k: move_tensor(v) for k, v in targets.items()}
            #if isinstance(batch, dict):
            #    batch = {k: move_tensor(v) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            #elif isinstance(batch, (list, tuple)):
            #    batch = type(batch)(move_tensor(v) if isinstance(v, torch.Tensor) else v for v in batch)
            # move and cast batch
            #batch = {k: (v.to(device).float() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            #optimizer.zero_grad()
            # autocast forward
            with amp.autocast():
                #features = agent.compute_features(batch)
                #preds = agent.forward(features)
                #losses = agent.compute_loss(features, batch, preds)
                #loss = sum(losses.values())
                #features = batch[0]  # assuming batch = [features, targets]
                #targets = batch[1]
                preds = agent.forward(features, targets)
                #preds = agent.forward(batch)
                losses = agent.compute_loss(features, targets, preds)
                loss = losses["loss"]#sum(losses.values())
                #print(losses)
            # scale and backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_iter.set_postfix({"loss": f"{loss.item():.4f}"})	
            #optimizer.zero_grad()
        if is_main_process():
            logger.info(f"Epoch {epoch} training loss: {loss.item():.4f}")

        # validation
        model.eval()
        val_losses = []
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch} [Eval]", leave=False)
        with torch.no_grad():
            for features, targets in val_iter:
                features = move_to_device(features)
                targets = move_to_device(targets)
                with amp.autocast():
                    preds = agent.forward(features)
                    losses = agent.compute_loss(features, targets, preds)
                    val_losses.append(losses["loss"].item())
                val_iter.set_postfix({"val_loss": f"{val_losses[-1]:.4f}"})
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('nan')
        if is_main_process():
            logger.info(f"Epoch {epoch} validation loss: {avg_val_loss:.4f}")
            ckpt_dir = "/home/zhengmi/DiffusionDrive/ckpts/navsim"#getattr(cfg, 'output_dir', '.')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")



if __name__ == "__main__":
    main()

