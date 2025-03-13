sweep_config = {
    # Base model
    'checkpoint_path': "/lustre/groups/ml01/projects/2023_nicheformer/pretrained_models/everything_heads_16_blocks_12_maxsteps_30661140_FINAL/epoch=1-step=265000.ckpt",

    # Data
    'data_path': "/lustre/groups/ml01/projects/2023_nicheformer_data_anna.schaar/spatial/preprocessed/human/nanostring_cosmx_human_liver.h5ad",
    'technology_mean_path': "/lustre/groups/ml01/projects/2023_nicheformer/data/data_to_tokenize/cosmx_mean_script.npy",
    'sae_hidden_dim': 2048,
    'sae_l1_coefficient': 0.1,
    'sae_tied_weights': False,
    'sae_activation': 'relu',
    'sae_layers': None,

    'freeze_base_model': True,
    'lr': 1e-3,
    'warmup': 1000,
    'max_epochs': 10000,
    'batch_size': 12,
    'num_workers': 4,
    'precision': 32,
    'gradient_clip_val': 1.0,

    'output_dir': './outputs/sae_training',
    'run_name': 'nicheformer_sae',
    'organ': 'brain',
} 