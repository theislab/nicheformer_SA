import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for feature extraction from transformer layers.
    
    Implements a sparse autoencoder that can be applied to the outputs of 
    transformer layers to extract interpretable features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        l1_coefficient: float = 0.1,
        tied_weights: bool = False,
        activation: str = "relu",
        bias: bool = True,
    ):
        """
        Initialize the sparse autoencoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of the hidden (code) layer
            l1_coefficient: Coefficient for L1 sparsity penalty
            tied_weights: If True, decoder weights are tied to encoder weights
            activation: Activation function to use ('relu', 'gelu', or 'sigmoid')
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coefficient = l1_coefficient
        self.tied_weights = tied_weights
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=bias)
        
        # Decoder (if weights are tied, we'll use the encoder's weights transposed)
        if not tied_weights:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=bias)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to hidden representation."""
        return self.activation(self.encoder(x))
    
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode hidden representation to reconstruction."""
        if self.tied_weights:
            # Use transposed encoder weights for decoding
            return F.linear(h, self.encoder.weight.t(), 
                           self.encoder.bias if self.encoder.bias is not None else None)
        else:
            return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, hidden_representation)
        """
        # Encode
        h = self.encode(x)
        
        # Decode
        reconstruction = self.decode(h)
        
        return reconstruction, h
    
    def loss(self, x: torch.Tensor, reconstruction: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with reconstruction error and sparsity penalty.
        
        Args:
            x: Original input
            reconstruction: Reconstructed input
            hidden: Hidden layer activations
            
        Returns:
            Total loss (reconstruction loss + sparsity penalty)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)
        
        # L1 sparsity penalty
        l1_penalty = self.l1_coefficient * hidden.abs().mean()
        
        return recon_loss + l1_penalty 