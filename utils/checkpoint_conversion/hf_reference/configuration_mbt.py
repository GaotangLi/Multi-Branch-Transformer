"""Hugging Face configuration for Multi-Branch Transformer."""
from transformers import PretrainedConfig


class MBTConfig(PretrainedConfig):
    model_type = "mbt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=64,
        moe_latent_size=512,
        moe_intermediate_size=512,
        num_experts=64,
        num_experts_per_tok=8,
        router_renorm_sum=2.5,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        prelude=0,
        coda=0,
        branch_depths=None,
        branch_hidden_size=None,
        branch_moe_latent_size=None,
        branch_moe_intermediate_size=None,
        branch_num_attention_heads=None,
        branch_num_key_value_heads=None,
        kv_mode="standard",
        use_input_bridge=False,
        initializer_range=0.02,
        max_position_embeddings=4096,
        use_cache=True,
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=128254,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id,
            pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.moe_latent_size = moe_latent_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_renorm_sum = router_renorm_sum
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.prelude = prelude
        self.coda = coda
        self.branch_depths = list(branch_depths or [])
        self.branch_hidden_size = branch_hidden_size if branch_hidden_size is not None else hidden_size
        self.branch_moe_latent_size = branch_moe_latent_size if branch_moe_latent_size is not None else moe_latent_size
        self.branch_moe_intermediate_size = branch_moe_intermediate_size if branch_moe_intermediate_size is not None else moe_intermediate_size
        self.branch_num_attention_heads = branch_num_attention_heads if branch_num_attention_heads is not None else num_attention_heads
        self.branch_num_key_value_heads = branch_num_key_value_heads if branch_num_key_value_heads is not None else num_key_value_heads
        self.kv_mode = kv_mode
        self.use_input_bridge = use_input_bridge
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        if not tie_word_embeddings:
            raise ValueError("Released MBT checkpoints use tied input/output embeddings")
