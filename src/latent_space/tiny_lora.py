"""
Implementation of TinyLoRA from:
Morris, J. X., Mireshghallah, N., Ibrahim, M., & Mahloujifar, S. (2026).
"Learning to Reason in 13 Parameters."
arXiv preprint arXiv:2602.04118
https://arxiv.org/abs/2602.04118
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import List

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

        # Freeze and store the base weights
        self.weight = base_layer.weight
        self.weight.requires_grad = False
        if base_layer.bias is not None:
            self.bias = base_layer.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        # Compute truncated SVD of the frozen weight matrix (W)
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

        # Initialize fixed random tensor P of shape (u, r, r)
        # Using a normal distribution to initialize the random matrices
        P = torch.randn(u, r, r)
        self.register_buffer('P', P)

        # Initialize the trainable vector v
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


def apply_tiny_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 2,
    u: int = 1,
    tie_all_layers: bool = True
) -> nn.Module:
    """
    Replace specified linear layers in any PyTorch model with TinyLoRALinear.

    Args:
        model: The base PyTorch model.
        target_modules: A list of string fragments to match layer names
                        (e.g., ["q_proj", "v_proj", "fc1"]).
        r: The rank for the frozen SVD components.
        u: The dimension of the trainable vector v.
        tie_all_layers: If True, all adapted modules share a single trainable `v`
                        parameter, minimizing total parameter count.
    """
    shared_v = nn.Parameter(torch.zeros(u)) if tie_all_layers else None

    # Collect paths to target layers without modifying the dict during iteration
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if any target string is in the current module's name
            if any(target in name for target in target_modules):
                modules_to_replace.append(name)

    # Swap the target linear layers with TinyLoRALinear
    for name in modules_to_replace:
        # Split the path to get the parent module and the specific child attribute
        parent_path = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]

        # Retrieve the parent module
        if parent_path == "":
            parent = model
        else:
            parent = model.get_submodule(parent_path)

        target_layer = getattr(parent, child_name)

        # For LinearKMaskedBias (e.g. DINOv3's fused QKV layer), the K-bias
        # mask is applied dynamically in its forward(). Bake it permanently
        # into the bias tensor now so the frozen base path inside
        # TinyLoRALinear stays correct without needing the custom forward.
        if hasattr(target_layer, "bias_mask") and target_layer.bias is not None:
            mask = target_layer.bias_mask.to(target_layer.bias.dtype).nan_to_num(nan=0.0)
            target_layer.bias.data.mul_(mask)

        # Wrap the layer
        new_layer = TinyLoRALinear(target_layer, r=r, u=u, shared_v=shared_v)

        # Hot-swap the layer inside the parent module
        setattr(parent, child_name, new_layer)

    # Register the shared parameter to the root model if tying layers
    if tie_all_layers and shared_v is not None:
        model.register_parameter("tiny_lora_v", shared_v)

    return model

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    # curl -LsSf https://hf.co/cli/install.sh | bash
    # uvx hf auth login

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Define target modules
    targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # Apply TinyLoRA
    model = apply_tiny_lora(model, target_modules=targets, r=2, u=1, tie_all_layers=True)

    # Verify parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params}")
