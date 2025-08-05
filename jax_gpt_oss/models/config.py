"""Configuration for GPT-OSS-20B model."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json
from pathlib import Path


@dataclass
class GPTOSSConfig:
    """Configuration for GPT-OSS-20B model.
    
    Based on the official config from openai/gpt-oss-20b.
    """
    
    # Model dimensions
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_hidden_layers: int = 24
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    
    # MoE configuration
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    experts_per_token: int = 4
    router_aux_loss_coef: float = 0.9
    output_router_logits: bool = False
    
    # Attention configuration
    attention_bias: bool = True
    attention_dropout: float = 0.0
    sliding_window: int = 128
    max_position_embeddings: int = 131072
    initial_context_length: int = 4096
    
    # Layer types (alternating sliding/full attention)
    layer_types: List[str] = None
    
    # Activation and normalization
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    swiglu_limit: float = 7.0
    
    # RoPE configuration
    rope_theta: float = 150000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Model behavior
    use_cache: bool = True
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    
    # Special tokens
    eos_token_id: int = 200002
    pad_token_id: int = 199999
    bos_token_id: int = 199999
    
    # Data type
    dtype: str = "bfloat16"
    
    def __post_init__(self):
        """Initialize derived attributes."""
        if self.layer_types is None:
            # Default alternating pattern
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                if i % 2 == 0:
                    self.layer_types.append("sliding_attention")
                else:
                    self.layer_types.append("full_attention")
        
        if self.rope_scaling is None:
            self.rope_scaling = {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "rope_type": "yarn",
                "truncate": False
            }
    
    @classmethod
    def from_json(cls, json_path: str) -> "GPTOSSConfig":
        """Load configuration from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        config_dict = self.__dict__.copy()
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()