# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepSeek V4 INT4 quantization support.

Handles checkpoints where MoE expert weights have been converted from
MXFP4 to symmetric INT4 (packed 2 per uint8) with per-group BF16 scales.
Non-expert weights are stored as BF16.
"""

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs


class DeepseekV4Int4Config(QuantizationConfig):
    """Config for DeepSeek V4 models with INT4-quantized MoE experts.

    Expert weights are symmetric INT4 (packed 2 per uint8, group_size=32)
    with per-group BF16 scales. Non-expert weights are BF16.
    """

    def __init__(self, group_size: int = 32) -> None:
        super().__init__()
        self.group_size = group_size
        self.weight_bits = 4
        self.pack_factor = 8 // self.weight_bits  # 2 values per byte

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v4_int4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeepseekV4Int4Config":
        group_size = config.get("group_size", 32)
        return cls(group_size=group_size)

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if user_quant == "deepseek_v4_int4":
            return "deepseek_v4_int4"
        if isinstance(hf_quant_cfg, dict) and hf_quant_cfg.get(
            "quant_method"
        ) == "deepseek_v4_int4":
            return "deepseek_v4_int4"
        model_type = getattr(hf_config, "model_type", None)
        expert_dtype = getattr(hf_config, "expert_dtype", None)
        if model_type == "deepseek_v4" and expert_dtype == "int4":
            return "deepseek_v4_int4"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            return DeepseekV4Int4MoEMethod(
                moe=layer.moe_config, group_size=self.group_size
            )
        return None


class DeepseekV4Int4MoEMethod(FusedMoEMethodBase):
    """INT4 W4A16 MoE method for DeepSeek V4 experts.

    Uses the existing fused_experts INT4 kernel path with per-group scales.
    """

    def __init__(self, moe: FusedMoEConfig, group_size: int = 32) -> None:
        super().__init__(moe)
        self.group_size = group_size
        self.pack_factor = 2  # 2 INT4 values per uint8

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition

        # Fused gate_up_proj: w1 (gate) + w3 (up), packed INT4
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj: w2, packed INT4
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Per-group scales for w13
        w13_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        w13_scales.quant_method = "group"

        # Per-group scales for w2
        w2_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        w2_scales.quant_method = "group"

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size],
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # Convert signed int4 (two's complement stored as val & 0x0F) to
        # unsigned offset binary (val + 8) expected by the WNA16 kernel
        # with implicit zero_point=8.
        for attr in ("w13_weight", "w2_weight"):
            w = getattr(layer, attr).data
            low = w & 0x0F
            high = (w >> 4) & 0x0F
            low = (low + 8) & 0x0F
            high = (high + 8) & 0x0F
            w.copy_(low | (high << 4))

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.apply(
            layer=layer,
            x=x,
            topk_weights=router_logits,
            topk_ids=router_logits,
            shared_experts=None,
            shared_experts_input=None,
        )
