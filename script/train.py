import torch
import yaml
import datetime
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# set tf to cpu only
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import jax
jax.config.update("jax_platform_name", "cpu")

from vbd.data.dataset import WaymaxDataset
from vbd.model.VBD import VBD
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from matplotlib import pyplot as plt


def load_config(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def train(cfg):
    print("Start Training")
    
    pl.seed_everything(cfg["seed"])
    torch.set_float32_matmul_precision("high")    
        
    # create dataset
    train_dataset = WaymaxDataset(
        data_dir = cfg["train_data_path"],
        anchor_path=cfg["anchor_path"],
        # max_object= cfg["agents_len"],
    )
    
    val_dataset = WaymaxDataset(
        cfg["val_data_path"],
        anchor_path=cfg["anchor_path"],
        # max_object= cfg["agents_len"],
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["batch_size"], 
        pin_memory=True, 
        num_workers=cfg["num_workers"],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg["batch_size"],
        pin_memory=True, 
        num_workers=cfg["num_workers"],
        shuffle=False
    )
    
    output_root = cfg.get("log_dir", "output")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"{cfg['model_name']}_{timestamp}"
    output_path = f"{output_root}/{model_name}"
    print("Save to ", output_path)
    
    os.makedirs(output_path, exist_ok=True)
    # dump cfg to yaml file
    with open(f"{output_path}/config.yaml", "w") as file:
        yaml.dump(cfg, file)
    
    num_gpus = torch.cuda.device_count()
    print("Total GPUS:", num_gpus)
    model = VBD(cfg=cfg)

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        print("Load Weights from ", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu"))["state_dict"])
    
    if not cfg.get("train_encoder"):
        # load 
        encoder_path = cfg.get("encoder_ckpt", None)
        if encoder_path is not None:
            model_dict = torch.load(encoder_path, map_location=torch.device("cpu"))["state_dict"]
            for key in list(model_dict.keys()):
                if not key.startswith("encoder."):
                    del model_dict[key]
            # load parameters to model
            print("Load Encoder Weights")
            model.load_state_dict(model_dict, strict=False)
        else:
            cfg["train_encoder"] = True
            raise Warning("Encoder path is not provided")

    # Plot Scheduler
    plt.plot(model.noise_scheduler.alphas_cumprod.cpu().numpy())
    plt.plot(f"{output_path}/scheduler.jpg")
    plt.close()
    
    use_wandb = cfg.get("use_wandb", True)
    if use_wandb:
        logger = WandbLogger(
            name=model_name,
            project=cfg.get("project"),
            entity=cfg.get("username"),
            log_model=False,
            dir=output_path,
        )
    else:
        logger = CSVLogger(output_path, name="VBD", version=1, flush_logs_every_n_steps=100)
    
    trainer = pl.Trainer(
        num_nodes=cfg.get("num_nodes", 1),
        max_epochs=cfg["epochs"],
        devices=cfg.get("num_gpus", -1),
        accelerator="gpu",
        strategy= DDPStrategy() if num_gpus > 1 else "auto",
        enable_progress_bar=True, 
        logger=logger, 
        enable_model_summary=True,
        detect_anomaly=False,
        gradient_clip_val=1.0,  
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        log_every_n_steps=100,
        callbacks=[
            ModelCheckpoint(
                dirpath=output_path,
                save_top_k=20,
                save_weights_only=False,
                monitor="val/loss",
                filename="epoch={epoch:02d}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval="step")
        ]
    )
    print("Build Trainer")
    
    trainer.fit(
        model, 
        train_loader, 
        val_loader, 
        ckpt_path=cfg.get("init_from")
    )
    
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", type=str, default="config/VBD.yaml")
    
    # Params for override config
    parser.add_argument("-name", "--model_name", type=str, default=None)
    parser.add_argument("-log", "-log_dir", type=str, default=None)
    
    parser.add_argument("-step", "--diffusion_steps", type=int, default=None)
    parser.add_argument("-mean", "--action_mean", nargs=2, metavar=("accel", "yaw"),
                        type=float, default=None)
    parser.add_argument("-std", "--action_std", nargs=2, metavar=("accel", "yaw"),
                        type=float, default=None)
    parser.add_argument("-zD", "--embeding_dim", type=int, default=None)
    parser.add_argument("-clamp", "--clamp_value", type=float, default=None)
    parser.add_argument("-init", "--init_from", type=str, default=None)
    parser.add_argument("-encoder", "--encoder_ckpt", type=str, default=None)
    parser.add_argument("-nN", "--num_nodes", type=int, default=1)
    parser.add_argument("-nG", "--num_gpus", type=int, default=-1)
    parser.add_argument("-sType", "--schedule_type", type=str, default=None)
    parser.add_argument("-sS", "--schedule_s", type=float, default=None)
    parser.add_argument("-sE", "--schedule_e", type=float, default=None)
    parser.add_argument("-scale", "--schedule_scale", type=float, default=None)
    parser.add_argument("-sT", "--schedule_tau", type=float, default=None)
    parser.add_argument("-eV", "--encoder_version", type=str, default=None)
    parser.add_argument("-pred", "--with_predictor", type=bool, default=None)
    parser.add_argument("-type", "--prediction_type", type=str, default=None)
    
    return parser
    
def load_cfg(args):
    cfg = load_config(args.cfg)
    
    # Override config from args
    # Iterate the args and override the config
    for key, value in vars(args).items():
        if key == "cfg":
            pass
        elif value is not None:
            cfg[key] = value
    return cfg
    
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_cfg(args)
    
    train(cfg)
    
