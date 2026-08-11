# Copyright 2026 Bytedance Ltd. and/or its affiliates.
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

"""CPU tests for the phase-2 MoE-LoRA adapter key split in the automodel engine.

Phase-2 LoRA rollout weight sync sends Automodel's fused 3D
``GroupedExpertsLoRA`` adapter params (``lora_gate_and_up_A/B`` fused w1+w3,
``lora_down_A/B`` = w2) to vLLM ``add_lora``. vLLM expects per-expert
``<prefix>.{i}.{gate_proj,up_proj,down_proj}.lora_{A,B}.weight`` 2D tensors
that ``PackedLoRALayerWeights.pack_moe`` regroups into the ``[w1, w2, w3]``
lists ``FusedMoEWithLoRA.set_lora`` consumes. These tests pin that mapping.
"""

import torch

from verl.workers.engine.automodel.utils import (
    _MoELoRASpec,
    split_moe_lora_adapter,
)


def _spec(n_inter=2048, gated=True):
    return _MoELoRASpec(prefix="model.layers.1.mlp.experts", moe_inter_dim=n_inter, is_gated=gated)


def test_split_gate_and_up_a_emits_w1_and_w3_a():
    """lora_gate_and_up_A [n, expert_dim, lora_dim] -> gate_proj.lora_A (w1) and
    up_proj.lora_A (w3), identical (shared input factor)."""
    n_exp, expert_dim, lora_dim = 3, 5120, 32
    t = torch.randn(n_exp, expert_dim, lora_dim)
    out = list(split_moe_lora_adapter(_spec(), "lora_gate_and_up_A", t))

    keys = [k for k, _ in out]
    assert keys == [
        "model.layers.1.mlp.experts.0.gate_proj.lora_A.weight",
        "model.layers.1.mlp.experts.0.up_proj.lora_A.weight",
        "model.layers.1.mlp.experts.1.gate_proj.lora_A.weight",
        "model.layers.1.mlp.experts.1.up_proj.lora_A.weight",
        "model.layers.1.mlp.experts.2.gate_proj.lora_A.weight",
        "model.layers.1.mlp.experts.2.up_proj.lora_A.weight",
    ]
    # w1 A and w3 A are [in, rank] and identical (shared input).
    for i in range(n_exp):
        w1a = dict(out)[f"model.layers.1.mlp.experts.{i}.gate_proj.lora_A.weight"]
        w3a = dict(out)[f"model.layers.1.mlp.experts.{i}.up_proj.lora_A.weight"]
        assert w1a.shape == (expert_dim, lora_dim)
        assert w3a.shape == (expert_dim, lora_dim)
        torch.testing.assert_close(w1a, t[i])
        torch.testing.assert_close(w3a, t[i])
        # w3 A must be a clone, not a view (add_lora may move/keep it).
        assert w3a.data_ptr() != w1a.data_ptr()


def test_split_gate_and_up_b_emits_w1_and_w3_b_halves():
    """lora_gate_and_up_B [n, lora_dim, 2*moe_inter_dim] -> gate_proj.lora_B (w1,
    first half) and up_proj.lora_B (w3, second half), each [rank, moe_inter_dim]."""
    n_exp, lora_dim, moe_inter_dim = 3, 32, 2048
    t = torch.randn(n_exp, lora_dim, 2 * moe_inter_dim)
    out = dict(split_moe_lora_adapter(_spec(n_inter=moe_inter_dim), "lora_gate_and_up_B", t))

    for i in range(n_exp):
        w1b = out[f"model.layers.1.mlp.experts.{i}.gate_proj.lora_B.weight"]
        w3b = out[f"model.layers.1.mlp.experts.{i}.up_proj.lora_B.weight"]
        assert w1b.shape == (lora_dim, moe_inter_dim)
        assert w3b.shape == (lora_dim, moe_inter_dim)
        torch.testing.assert_close(w1b, t[i, :, :moe_inter_dim])
        torch.testing.assert_close(w3b, t[i, :, moe_inter_dim:])


def test_split_down_a_emits_w2_a():
    """lora_down_A [n, moe_inter_dim, lora_dim] -> down_proj.lora_A (w2 A)
    [moe_inter_dim, lora_dim]."""
    n_exp, moe_inter_dim, lora_dim = 2, 2048, 32
    t = torch.randn(n_exp, moe_inter_dim, lora_dim)
    out = dict(split_moe_lora_adapter(_spec(n_inter=moe_inter_dim), "lora_down_A", t))

    for i in range(n_exp):
        w2a = out[f"model.layers.1.mlp.experts.{i}.down_proj.lora_A.weight"]
        assert w2a.shape == (moe_inter_dim, lora_dim)
        torch.testing.assert_close(w2a, t[i])


def test_split_down_b_emits_w2_b():
    """lora_down_B [n, lora_dim, expert_dim] -> down_proj.lora_B (w2 B)
    [lora_dim, expert_dim]."""
    n_exp, lora_dim, expert_dim = 2, 32, 5120
    t = torch.randn(n_exp, lora_dim, expert_dim)
    out = dict(split_moe_lora_adapter(_spec(), "lora_down_B", t))

    for i in range(n_exp):
        w2b = out[f"model.layers.1.mlp.experts.{i}.down_proj.lora_B.weight"]
        assert w2b.shape == (lora_dim, expert_dim)
        torch.testing.assert_close(w2b, t[i])


def test_split_non_gated_skips_w3():
    """For non-gated MoE w3 is unused (pack_moe aliases w3 to w1); up_proj keys
    are skipped."""
    n_exp, expert_dim, lora_dim = 2, 5120, 32
    # gate_and_up_B is [n, lora_dim, moe_inter_dim] (no doubled up_proj_dim) when
    # non-gated; w3 must not be emitted.
    moe_inter_dim = 2048
    a = torch.randn(n_exp, expert_dim, lora_dim)
    b = torch.randn(n_exp, lora_dim, moe_inter_dim)

    out_a = list(split_moe_lora_adapter(_spec(n_inter=moe_inter_dim, gated=False), "lora_gate_and_up_A", a))
    out_b = list(split_moe_lora_adapter(_spec(n_inter=moe_inter_dim, gated=False), "lora_gate_and_up_B", b))

    assert all("up_proj" not in k for k, _ in out_a)
    assert all("up_proj" not in k for k, _ in out_b)
    assert any(k.endswith("gate_proj.lora_A.weight") for k, _ in out_a)
    assert any(k.endswith("gate_proj.lora_B.weight") for k, _ in out_b)


def test_split_unknown_attr_raises():
    t = torch.randn(1, 2, 3)
    try:
        list(split_moe_lora_adapter(_spec(), "lora_unknown", t))
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown adapter attr")


def test_to_vllm_peft_dict_match_all_linear_passes_none():
    """match_all_linear=True -> target_modules=None (vLLM treats None as 'all';
    an empty list means 'nothing')."""
    from verl.workers.engine.automodel.utils import to_vllm_peft_dict

    class _Cfg:
        def to_dict(self):
            return {"dim": 32, "alpha": 64, "match_all_linear": True, "use_dora": False}

    d = to_vllm_peft_dict(_Cfg())
    assert d["target_modules"] is None
    assert d["r"] == 32
    assert d["lora_alpha"] == 64
    assert d["use_dora"] is False


def test_to_vllm_peft_dict_explicit_target_modules_passes_list():
    """Explicit target_modules -> pass as a list (no match_all_linear)."""
    from verl.workers.engine.automodel.utils import to_vllm_peft_dict

    class _Cfg:
        def to_dict(self):
            return {"dim": 8, "alpha": 16, "match_all_linear": False, "use_dora": True}

    d = to_vllm_peft_dict(_Cfg())
    assert d["target_modules"] == []
    assert d["use_dora"] is True


def test_collect_moe_lora_prefixes_finds_grouped_experts_lora():
    """collect_automodel_lora_param_maps maps all four adapter attrs of every
    GroupedExpertsLoRA module to a spec with the right moe_inter_dim/is_gated."""
    from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA

    from verl.workers.engine.automodel.utils import collect_automodel_lora_param_maps

    class _Cfg:
        # GroupedExpertsLoRA._init_adapter reads obj.config.moe_inter_dim and
        # obj.is_gated; we only exercise collect_automodel_lora_param_maps (which
        # reads module.config.moe_inter_dim and module.is_gated), so set both.
        moe_inter_dim = 16

    # Bypass the heavy GroupedExperts.__init__ (allocates real params, needs a
    # full MoEConfig) — the collector only inspects isinstance + module.config /
    # module.is_gated. Still init nn.Module so named_modules works.
    lora_experts = GroupedExpertsLoRA.__new__(GroupedExpertsLoRA)
    torch.nn.Module.__init__(lora_experts)
    lora_experts.config = _Cfg()
    lora_experts.is_gated = True

    class _Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = lora_experts

    _, prefixes, _ = collect_automodel_lora_param_maps(_Holder())
    # state_dict keys for the holder are "experts.<attr>"
    expected = {
        "experts.lora_gate_and_up_A",
        "experts.lora_gate_and_up_B",
        "experts.lora_down_A",
        "experts.lora_down_B",
    }
    assert set(prefixes) == expected
    spec = prefixes["experts.lora_gate_and_up_A"]
    assert spec.prefix == "experts"
    assert spec.moe_inter_dim == 16
    assert spec.is_gated is True


def test_collect_moe_lora_prefixes_normalizes_checkpoint_wrapper():
    """The ``._checkpoint_wrapped_module.`` segment from activation checkpointing
    is normalized out so prefixes match state_dict() keys."""
    from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA

    from verl.workers.engine.automodel.utils import collect_automodel_lora_param_maps

    class _Cfg:
        moe_inter_dim = 16

    lora_experts = GroupedExpertsLoRA.__new__(GroupedExpertsLoRA)
    torch.nn.Module.__init__(lora_experts)
    lora_experts.config = _Cfg()
    lora_experts.is_gated = True

    class _ACWrapper(torch.nn.Module):
        """Mimics activation-checkpointing wrapping: the real module sits under
        ``_checkpoint_wrapped_module`` nested inside a real parent (as it is in
        practice — e.g. ``model.layers.1._checkpoint_wrapped_module.experts``)."""

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Module()
            self.layers._checkpoint_wrapped_module = torch.nn.Module()
            self.layers._checkpoint_wrapped_module.experts = lora_experts

    _, prefixes, _ = collect_automodel_lora_param_maps(_ACWrapper())
    # The ``._checkpoint_wrapped_module.`` segment is normalized out so the prefix
    # matches the state_dict() key the rest of the pipeline emits.
    assert "layers._checkpoint_wrapped_module.experts.lora_gate_and_up_A" not in prefixes
    assert "layers.experts.lora_gate_and_up_A" in prefixes
    assert prefixes["layers.experts.lora_gate_and_up_A"].prefix == "layers.experts"


# ---------------------------------------------------------------------------
# Phase-1: LoRA-wrapped MoE expert_params_mapping base_layer/routed_experts
# ordering. The legacy ``fused_moe_make_expert_params_mapping`` builds
# ``param_name = experts.routed_experts.base_layer.w13_`` but the live LoRA
# tree is ``experts.base_layer.routed_experts.w13_weight``. verl wraps the
# module-level function (see ``verl.utils.vllm.patch``) to reorder the
# ``param_name`` segments so the strict loader's ``params_dict[name_mapped]``
# resolves.
# ---------------------------------------------------------------------------


def _make_dsv2_model():
    """A fake CausalLM whose ``.model`` child is named ``DeepseekV2Model`` so the
    reorder patch's architecture gate fires (only DeepseekV2/DSV3 hit the bug)."""

    class DeepseekV2Model(torch.nn.Module):
        pass

    class _DSV2CausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DeepseekV2Model()

    return _DSV2CausalLM()


def test_patch_lora_moe_expert_mapping_order_reorders_param_name():
    """The patch swaps ``routed_experts.base_layer.`` -> ``base_layer.routed_experts.``
    in ``param_name`` for LoRA-wrapped DeepseekV2 MoE; the checkpoint
    ``weight_name`` is unchanged so the loader's substring match still fires."""
    import vllm.model_executor.layers.fused_moe as fused_moe

    from verl.utils.vllm.patch import _patch_lora_moe_expert_mapping_order

    # Save & restore the module-level function so the global patch doesn't leak
    # across tests.
    original = fused_moe.fused_moe_make_expert_params_mapping
    try:
        # Undo a possibly-already-applied wrapper so we exercise a clean apply.
        wrapped = getattr(original, "_verl_original", None)
        if wrapped is not None:
            fused_moe.fused_moe_make_expert_params_mapping = wrapped
        _patch_lora_moe_expert_mapping_order(_make_dsv2_model())
        patched = fused_moe.fused_moe_make_expert_params_mapping
        assert getattr(patched, "_verl_lora_order_patched", False)

        class _FakeMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.empty(0))

            def named_parameters(self, remove_duplicate=False):
                # Triggers the legacy ``has_base_layer`` detection.
                yield "experts.base_layer.routed_experts.w13_weight", self.p

        out = fused_moe.fused_moe_make_expert_params_mapping(_FakeMod(), "gate_proj", "down_proj", "up_proj", 2, 0)
        # param_name (field 0) reordered; weight_name (field 1) unchanged.
        for param_name, weight_name, _eid, _sid in out:
            assert "routed_experts.base_layer." not in param_name
            assert param_name.startswith("experts.base_layer.routed_experts.")
            assert ".base_layer." in weight_name  # checkpoint pattern intact
        # Idempotent: re-applying does not nest wrappers.
        _patch_lora_moe_expert_mapping_order(_make_dsv2_model())
        assert fused_moe.fused_moe_make_expert_params_mapping is patched
    finally:
        fused_moe.fused_moe_make_expert_params_mapping = original


def test_patch_lora_moe_expert_mapping_order_skipped_for_non_deepseek():
    """The reorder is gated to DeepseekV2/DSV3: a non-Deepseek model leaves the
    module-level function untouched (other MoE loaders don't hit the bug)."""
    import vllm.model_executor.layers.fused_moe as fused_moe

    from verl.utils.vllm.patch import _patch_lora_moe_expert_mapping_order

    original = fused_moe.fused_moe_make_expert_params_mapping
    try:
        wrapped = getattr(original, "_verl_original", None)
        if wrapped is not None:
            fused_moe.fused_moe_make_expert_params_mapping = wrapped
        pristine = fused_moe.fused_moe_make_expert_params_mapping

        class _OtherCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()  # not "DeepseekV2Model"

        _patch_lora_moe_expert_mapping_order(_OtherCausalLM())
        # No wrapper installed.
        assert fused_moe.fused_moe_make_expert_params_mapping is pristine
        assert not getattr(fused_moe.fused_moe_make_expert_params_mapping, "_verl_lora_order_patched", False)
    finally:
        fused_moe.fused_moe_make_expert_params_mapping = original


def test_patch_lora_moe_expert_mapping_order_passes_through_non_lora():
    """Non-LoRA mappings (no ``base_layer.`` in param_name) pass through
    unchanged -- the reorder only applies to LoRA-wrapped MoE."""
    import vllm.model_executor.layers.fused_moe as fused_moe

    from verl.utils.vllm.patch import _patch_lora_moe_expert_mapping_order

    original = fused_moe.fused_moe_make_expert_params_mapping
    try:
        wrapped = getattr(original, "_verl_original", None)
        if wrapped is not None:
            fused_moe.fused_moe_make_expert_params_mapping = wrapped
        _patch_lora_moe_expert_mapping_order(_make_dsv2_model())

        class _FakeMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.empty(0))

            def named_parameters(self, remove_duplicate=False):
                # No ``.base_layer.`` -> non-LoRA mapping.
                yield "experts.routed_experts.w13_weight", self.p

        out = fused_moe.fused_moe_make_expert_params_mapping(_FakeMod(), "gate_proj", "down_proj", "up_proj", 2, 0)
        for param_name, _weight_name, _eid, _sid in out:
            assert param_name == "experts.routed_experts.w13_" or param_name == "experts.routed_experts.w2_"
            assert "base_layer" not in param_name
    finally:
        fused_moe.fused_moe_make_expert_params_mapping = original
