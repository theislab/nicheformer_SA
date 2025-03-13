import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader

from nicheformer.models._nicheformer_sae_wrapper import NicheformerSAEWrapper
from nicheformer.data.dataset import NicheformerDataset
import anndata as ad
import numpy as np

def train_sparse_autoencoders(config):
    """
    Train sparse autoencoders on top of a pre-trained Nicheformer model.
    
    Args:
        config: Configuration dictionary with training parameters
    """
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Load data
    adata = ad.read_h5ad(config['data_path'])
    technology_mean = np.load(config['technology_mean_path'])
    
    # making data compatible with nicheformer 
    model = ad.read_h5ad('/lustre/groups/ml01/projects/2023_nicheformer/data/data_to_tokenize/model.h5ad')

    adata = ad.concat([model, adata], join='outer', axis=0)
    # dropping the first observation 
    adata = adata[1:].copy()

    # Set 15% of observations to validation split randomly
    n_samples = len(adata)
    n_val = int(0.15 * n_samples)

    val_indices = np.random.choice(n_samples, size=n_val, replace=False)
    val_mask = np.zeros(n_samples, dtype=bool)
    val_mask[val_indices] = True
    adata.obs['nicheformer_split'] = 'train'
    adata.obs.loc[val_mask, 'nicheformer_split'] = 'val'

    # Create datasets
    train_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split='train',
        max_seq_len=config.get('max_seq_len', 1500),
        aux_tokens=config.get('aux_tokens', 30),
        chunk_size=config.get('chunk_size', 1000),
        metadata_fields=config.get('metadata_fields', {
            'obs': ['author_cell_type'],
            'obsm': []
        })
    )
    
    val_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split='val',
        max_seq_len=config.get('max_seq_len', 1500),
        aux_tokens=config.get('aux_tokens', 30),
        chunk_size=config.get('chunk_size', 1000),
        metadata_fields=config.get('metadata_fields', {
            'obs': ['author_cell_type'],
            'obsm': []
        })
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model
    model = NicheformerSAEWrapper(
        checkpoint_path=config['checkpoint_path'],
        sae_hidden_dim=config.get('sae_hidden_dim', 2048),
        sae_l1_coefficient=config.get('sae_l1_coefficient', 0.1),
        sae_tied_weights=config.get('sae_tied_weights', False),
        sae_activation=config.get('sae_activation', 'relu'),
        sae_layers=config.get('sae_layers', None),
        freeze_base_model=config.get('freeze_base_model', True),
        lr=config.get('lr', 1e-3),
        warmup=config.get('warmup', 1000),
        max_epochs=config.get('max_epochs', 10000)
    )


    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Set up trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=config.get('devices', 1),
        max_epochs=config.get('max_epochs', 10000),
        callbacks=[lr_monitor],
        precision=config.get('precision', 32),
        gradient_clip_val=config.get('gradient_clip_val', 1.0)
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)


    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sparse autoencoders for Nicheformer")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Train SAEs
    train_sparse_autoencoders(config)