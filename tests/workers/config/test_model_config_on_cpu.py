# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright Amazon.com and/or its affiliates
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
import os

import pytest
from omegaconf import OmegaConf

from verl.workers.config.model import HFModelConfig


class TestHFModelConfigCPU:
    model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B")  # Just a path string, not loaded

    def test_target_modules_accepts_list_via_omegaconf(self):
        """
        Test that target_modules field accepts both string and list values
        when merging OmegaConf configs (simulates CLI override behavior).

        The purpose is to ensure we can pass
        actor_rollout_ref.model.target_modules='["k_proj","o_proj","down_proj","q_proj"]'
        """

        # Create structured config from the dataclass defaults
        # This is what omega_conf_to_dataclass does internally
        cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

        # Simulate CLI override with target_modules as a list
        cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": ["k_proj", "o_proj", "q_proj", "v_proj"],
            }
        )

        # This merge should NOT raise ValidationError
        # Before the fix (target_modules: str), this would fail with:
        # "Cannot convert 'ListConfig' to string"
        merged = OmegaConf.merge(cfg_from_dataclass, cli_config)

        # Verify the list was merged correctly
        assert list(merged.target_modules) == ["k_proj", "o_proj", "q_proj", "v_proj"]

    def test_target_modules_accepts_none_via_omegaconf(self):
        """Test that target_modules still accepts None values."""

        cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

        cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": None,
            }
        )

        merged = OmegaConf.merge(cfg_from_dataclass, cli_config)
        assert merged.target_modules is None

    def test_target_modules_accepts_string_via_omegaconf(self):
        """Test that target_modules still accepts string values."""

        cfg_from_dataclass = OmegaConf.structured(HFModelConfig)

        cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": "all-linear",
            }
        )

        merged = OmegaConf.merge(cfg_from_dataclass, cli_config)
        assert merged.target_modules == "all-linear"

    def test_target_modules_raises_on_invalid_type(self):
        """Test that __post_init__ raises TypeError for invalid target_modules types."""
        base_config = OmegaConf.structured(HFModelConfig)
        invalid_cli_config = OmegaConf.create(
            {
                "path": self.model_path,
                "target_modules": [1, 2, 3],  # list of ints instead of strings
            }
        )
        merged_config = OmegaConf.merge(base_config, invalid_cli_config)
        with pytest.raises(TypeError):
            OmegaConf.to_object(merged_config)

    def test_critic_parameter_freeze_fields_accept_plain_hydra_overrides(self):
        """The PPO config must expose freeze fields without ``+key`` overrides."""
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("verl/trainer/config")
        manifest_path = "/tmp/critic-parameter-freeze.json"
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(
                config_name="ppo_trainer",
                overrides=[
                    "critic.model.parameter_freeze_policy=qwen35_dense_token_mixers_v1",
                    f"critic.model.parameter_freeze_manifest_path={manifest_path}",
                    "critic.model.parameter_freeze_probe_elements_per_rank=4096",
                ],
            )

        assert config.critic.model.parameter_freeze_policy == "qwen35_dense_token_mixers_v1"
        assert config.critic.model.parameter_freeze_manifest_path == manifest_path
        assert config.critic.model.parameter_freeze_probe_elements_per_rank == 4096
