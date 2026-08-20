# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace
import unittest

import torch

from verl.workers.engine.fsdp.transformer_impl import _liger_target_model


class TestFSDPLigerValueHead(unittest.TestCase):
    def test_liger_targets_the_pretrained_model_inside_a_value_head(self):
        base_model = object()
        value_head = SimpleNamespace(pretrained_model=base_model)

        self.assertIs(
            _liger_target_model(value_head, model_type="value_model"), base_model
        )
        self.assertIs(
            _liger_target_model(base_model, model_type="language_model"), base_model
        )

    def test_liger_value_model_without_pretrained_model_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "pretrained_model"):
            _liger_target_model(object(), model_type="value_model")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_qwen35_value_head_liger_forward(self):
        try:
            from liger_kernel.transformers.monkey_patch import (
                _apply_liger_kernel_to_instance,
            )
            from transformers.models.qwen3_5.configuration_qwen3_5 import (
                Qwen3_5TextConfig,
            )
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5ForCausalLM,
            )
            from trl import AutoModelForCausalLMWithValueHead
        except ImportError as exc:
            self.skipTest(str(exc))

        from verl.models.transformers.monkey_patch import apply_monkey_patch

        config = Qwen3_5TextConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            layer_types=["full_attention"],
            use_cache=False,
        )
        base_model = Qwen3_5ForCausalLM(config).to(
            device="cuda", dtype=torch.bfloat16
        )
        value_model = AutoModelForCausalLMWithValueHead(base_model).to(
            device="cuda", dtype=torch.bfloat16
        )
        # Direct construction omits this flag; from_pretrained sets it.
        value_model.is_peft_model = False
        target = _liger_target_model(value_model, model_type="value_model")
        _apply_liger_kernel_to_instance(
            model=target,
            fused_linear_cross_entropy=False,
            swiglu=True,
        )
        apply_monkey_patch(
            model=value_model,
            use_remove_padding=False,
            use_fused_kernels=False,
        )
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits, _, values = value_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        self.assertEqual(tuple(logits.shape), (2, 32, config.vocab_size))
        self.assertEqual(tuple(values.shape), (2, 32))


if __name__ == "__main__":
    unittest.main()
