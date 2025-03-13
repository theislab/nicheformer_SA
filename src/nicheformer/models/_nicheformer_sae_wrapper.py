import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Optional, Dict, Any, Union
import os
import copy

from ._sparse_autoencoder import SparseAutoencoder
from ._nicheformer import Nicheformer
from ._utils import complete_masking


class NicheformerSAEWrapper(pl.LightningModule):
    """
    Wrapper for adding sparse autoencoders to a pre-trained Nicheformer model.
    
    This class loads a pre-trained Nicheformer model and adds sparse autoencoders
    to specified layers. The base model can be kept frozen while training only
    the sparse autoencoders.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        sae_hidden_dim: int = 2048,
        sae_l1_coefficient: float = 0.1,
        sae_tied_weights: bool = False,
        sae_activation: str = "relu",
        sae_layers: Optional[List[int]] = None,
        freeze_base_model: bool = True,
        lr: float = 1e-3,
        warmup: int = 1000,
        max_epochs: int = 10000,
    ):
        """
        Initialize the wrapper.
        
        Args:
            checkpoint_path: Path to the pre-trained Nicheformer checkpoint
            sae_hidden_dim: Hidden dimension for sparse autoencoders
            sae_l1_coefficient: L1 coefficient for sparsity penalty
            sae_tied_weights: Whether to use tied weights in autoencoders
            sae_activation: Activation function for autoencoders
            sae_layers: List of layer indices to apply sparse autoencoders to (None = all layers)
            freeze_base_model: Whether to freeze the base model parameters
            lr: Learning rate for training the sparse autoencoders
            warmup: Number of warmup steps
            max_epochs: Maximum number of epochs
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load the pre-trained model
        self.base_model = Nicheformer.load_from_checkpoint(checkpoint_path)
        
        # Freeze the base model if requested
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Determine which layers to add SAEs to
        if sae_layers is None:
            sae_layers = list(range(len(self.base_model.encoder.layers)))
        self.sae_layers = sae_layers
        
        # Create sparse autoencoders for specified layers
        self.sparse_autoencoders = nn.ModuleList([
            SparseAutoencoder(
                input_dim=self.base_model.hparams.dim_model,
                hidden_dim=sae_hidden_dim,
                l1_coefficient=sae_l1_coefficient,
                tied_weights=sae_tied_weights,
                activation=sae_activation
            ) if i in sae_layers else None
            for i in range(len(self.base_model.encoder.layers))
        ])
        
        self.gc_freq = 5 # not needed now 
        self.batch_train_losses = []
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            batch: Input batch containing 'X' and other required fields
            
        Returns:
            Dictionary with model outputs and SAE features
        """
        # Prepare input for the base model
        batch_processed = complete_masking(batch, 0.0, self.base_model.hparams.n_tokens) # masking 0.0? change if needed, default is 0.15
        masked_indices = batch_processed['masked_indices'] # it is not really masked since masking_p = 0.0
        attention_mask = batch_processed['attention_mask']
        
        # Get token embeddings
        token_embedding = self.base_model.embeddings(masked_indices)
        
        # Add positional embeddings
        if self.base_model.hparams.learnable_pe:
            pos_embedding = self.base_model.positional_embedding(
                self.base_model.pos.to(token_embedding.device)
            )
            embeddings = self.base_model.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.base_model.positional_embedding(token_embedding)
        
        # Process through transformer layers and apply SAEs
        layer_outputs = []
        sae_outputs = []
        sae_hidden = []
        sae_losses = []

        current_embeddings = embeddings

        for i, layer in enumerate(self.base_model.encoder.layers):
            # Apply transformer layer
            current_embeddings = layer(
                current_embeddings,
                is_causal=False, # it is not gpt-like
                src_key_padding_mask=attention_mask
            )

            # Store layer output
            layer_outputs.append(current_embeddings) # shape: bs x n x d

            # Apply sparse autoencoder if enabled for this layer
            if i in self.sae_layers:
                # Reshape for autoencoder (combine batch and sequence dimensions)
                batch_size, seq_len, dim = current_embeddings.shape
                flat_embeddings = current_embeddings.reshape(-1, dim) # shape: bs * n x d not sure about this step

                # Apply autoencoder
                reconstruction, hidden = self.sparse_autoencoders[i](flat_embeddings)

                # Calculate loss
                sae_loss = self.sparse_autoencoders[i].loss(flat_embeddings, reconstruction, hidden)

                # Store outputs (i think this might explode in memory?)
                sae_outputs.append(reconstruction.reshape(batch_size, seq_len, dim))
                sae_hidden.append(hidden.reshape(batch_size, seq_len, -1))
                sae_losses.append(sae_loss)
            else:
                sae_outputs.append(None)
                sae_hidden.append(None)
                sae_losses.append(None)

        # Final transformer output is the last layer's output
        transformer_output = layer_outputs[-1]

        # Apply classifier head for MLM prediction
        mlm_prediction = self.base_model.classifier_head(transformer_output)

        return {
            'mlm_prediction': mlm_prediction,
            'transformer_output': transformer_output,
            'layer_outputs': layer_outputs,
            'sae_outputs': sae_outputs,
            'sae_hidden': sae_hidden,
            'sae_losses': sae_losses
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for the sparse autoencoders.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Total loss
        """
        # Forward pass
        outputs = self.forward(batch)

        # Calculate SAE losses
        sae_losses = [loss for loss in outputs['sae_losses'] if loss is not None]

        if not sae_losses:
            raise ValueError("No sparse autoencoders were applied. Check sae_layers configuration.")
        
        # Average SAE losses
        total_loss = torch.stack(sae_losses).mean()
        
        # Log metrics
        self.log('train_sae_loss', total_loss, sync_dist=True, prog_bar=True)
        
        # Store for later analysis
        self.batch_train_losses.append(total_loss.item())
        
        # Garbage collection
        if batch_idx % self.gc_freq == 0: # as said, not needed
            torch.cuda.empty_cache()
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for the sparse autoencoders.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Total loss
        """
        # Forward pass
        outputs = self.forward(batch)
        
        # Calculate SAE losses
        sae_losses = [loss for loss in outputs['sae_losses'] if loss is not None]
        
        if not sae_losses:
            raise ValueError("No sparse autoencoders were applied. Check sae_layers configuration.")
        
        # Average SAE losses
        total_loss = torch.stack(sae_losses).mean()
        
        # Log metrics
        self.log('val_sae_loss', total_loss, sync_dist=True)
        
        return total_loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer with constant learning rate.
        
        Returns:
            Optimizer with constant learning rate
        """
        # Only optimize SAE parameters
        optimizer = torch.optim.AdamW( 
            [p for p in self.parameters() if p.requires_grad], # parameters of base model are already frozen
            lr=self.hparams.lr,
            weight_decay=0.01
        )
        
        # Return only the optimizer for constant learning rate (no scheduler for SAE? idk)
        return optimizer