# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from verl.models.transformers.monkey_patch import patch_forward_with_backends


class _SharedModelClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="test_model")

    def forward(self):
        return "normal"


def _backend_module():
    module = ModuleType("verl.models.transformers.dense_common")

    def forward_with_torch_backend(self):
        return "torch-fused"

    def forward_with_triton_backend(self):
        return "triton-fused"

    module.forward_with_torch_backend = forward_with_torch_backend
    module.forward_with_triton_backend = forward_with_triton_backend
    return module


def test_fused_forward_is_scoped_to_configured_model_instance(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "verl.models.transformers.dense_common",
        _backend_module(),
    )
    actor = _SharedModelClass()
    critic = _SharedModelClass()
    original_class_forward = _SharedModelClass.forward

    patch_forward_with_backends(
        actor,
        use_fused_kernels=True,
        fused_kernels_backend="torch",
    )

    assert actor() == "torch-fused"
    assert critic() == "normal"
    assert _SharedModelClass.forward is original_class_forward
    assert "forward" in actor.__dict__
    assert "forward" not in critic.__dict__


def test_fused_qwen_actor_preserves_trl_critic_logits():
    trl = pytest.importorskip("trl")
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    text_config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    config = Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=27,
        vision_end_token_id=28,
    )
    actor = Qwen3_5ForConditionalGeneration(config)
    critic_backbone = Qwen3_5ForConditionalGeneration(config)
    original_class_forward = Qwen3_5ForConditionalGeneration.forward

    patch_forward_with_backends(
        actor,
        use_fused_kernels=True,
        fused_kernels_backend="torch",
    )

    assert Qwen3_5ForConditionalGeneration.forward is original_class_forward
    assert actor.forward.__func__.__name__ == "forward_with_torch_backend"
    assert critic_backbone.forward.__func__ is original_class_forward

    critic_backbone.config.hidden_size = config.text_config.hidden_size
    critic = trl.AutoModelForCausalLMWithValueHead.from_pretrained(critic_backbone)
    input_ids = torch.tensor([[1, 3, 4, 2]], dtype=torch.long)
    lm_logits, _, values = critic(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    assert lm_logits.shape == (1, 4, 32)
    assert values.shape == (1, 4)
