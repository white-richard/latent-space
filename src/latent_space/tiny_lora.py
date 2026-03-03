import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLoRALinear(nn.Module):
    def __init__(
        self, 
        base_layer: nn.Linear, 
        r: int = 2, 
        u: int = 1, 
        shared_v: nn.Parameter = None
    ):
        """
        TinyLoRA drop-in replacement for nn.Linear.
        
        Args:
            base_layer: The original pre-trained nn.Linear layer.
            r: The frozen rank for the truncated SVD. The paper recommends r=2.
            u: The trainable projection dimension.
            shared_v: An optional shared parameter vector to enable weight tying across layers.
        """
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.u = u
        
        # 1. Freeze and store the base weights
        self.weight = base_layer.weight
        self.weight.requires_grad = False
        if base_layer.bias is not None:
            self.bias = base_layer.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        # 2. Compute truncated SVD of the frozen weight matrix (W)
        with torch.no_grad():
            # W is shape (out_features, in_features)
            U, S, Vh = torch.linalg.svd(self.weight.float(), full_matrices=False)
            
            # Truncate to rank r
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]
            
        # Register U, Sigma, and V^T as frozen buffers
        self.register_buffer('U', U_r)
        self.register_buffer('Sigma', torch.diag(S_r))
        self.register_buffer('Vh', Vh_r)
        
        # 3. Initialize fixed random tensor P of shape (u, r, r)
        # Using a normal distribution to initialize the random matrices
        P = torch.randn(u, r, r)
        self.register_buffer('P', P)
        
        # 4. Initialize the trainable vector v
        if shared_v is not None:
            # Enable weight tying (sharing v across modules)
            self.v = shared_v
        else:
            # Standalone trainable vector v in R^u initialized to zero
            self.v = nn.Parameter(torch.zeros(u))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass: x @ W^T + bias
        base_out = F.linear(x, self.weight, self.bias)
        
        # TinyLoRA path: x @ (U \Sigma (\sum v_i P_i) V^T)^T
        # To save memory, we avoid realizing the full delta_W matrix 
        # and instead apply the matrices sequentially.
        
        # Step A: x @ V 
        x_V = F.linear(x, self.Vh)
        
        # Step B: Compute the combined r x r matrix R = \sum v_i P_i
        R = torch.einsum('i,ijk->jk', self.v, self.P)
        
        # Step C: Multiply by R^T
        x_VR = F.linear(x_V, R)
        
        # Step D: Multiply by \Sigma^T U^T
        # Note: Since Sigma is diagonal, Sigma^T = Sigma
        U_Sigma = self.U @ self.Sigma
        delta_out = F.linear(x_VR, U_Sigma)
        
        # Return the combined output
        return base_out + delta_out