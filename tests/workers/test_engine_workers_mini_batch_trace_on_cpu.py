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

"""How mini-batches show up in a trace of the engine workers.

The update loop names each iteration ``mini_batch<i>`` and advances the profiler once per
mini-batch: a mini-batch is the unit a ``torch.profiler.schedule`` sub-samples the update loop by
(see ``TorchProfilerScheduleConfig``). The forward-only stages (log-prob / ref / values) run their
batches without advancing the profiler, so a schedule never sub-samples them.

These tests drive the worker methods with mocked engines, so no GPU or ray is needed.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.trainer.ppo.core_algos import compute_policy_loss_bypass_mode
from verl.workers.engine_workers import TrainingWorker


class _StepHookOptimizer:
    def __init__(self):
        self._hooks = []

    def register_step_post_hook(self, hook):
        self._hooks.append(hook)

        def remove():
            self._hooks.remove(hook)

        return SimpleNamespace(remove=remove)

    def step(self):
        for hook in tuple(self._hooks):
            hook(self, (), {})


def _engine(**overrides):
    engine = SimpleNamespace(
        train_mode=lambda **kwargs: nullcontext(),
        eval_mode=lambda **kwargs: nullcontext(),
        infer_batch=lambda data, loss_function=None: {},
        is_mp_src_rank_with_outputs=lambda: False,
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_size=lambda: 1,
        optimizer=_StepHookOptimizer(),
    )
    for key, value in overrides.items():
        setattr(engine, key, value)
    return engine


def _worker(engine):
    return SimpleNamespace(
        engine=engine,
        engine_config=SimpleNamespace(
            forward_only=False,
            use_dynamic_bsz=False,
            infer_max_token_len_per_gpu=128,
            infer_micro_batch_size_per_gpu=1,
            max_token_len_per_gpu=128,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
        ),
        model_config={},
        optimizer_config=SimpleNamespace(
            lr_scheduler_step_per_optimizer_step=False
        ),
        loss_fn=lambda: None,
        profiler=MagicMock(),
    )


def _record_names(monkeypatch):
    """Collect the names of the record_function ranges the worker opens."""
    names = []

    def fake_record_function(name):
        names.append(name)
        return nullcontext()

    monkeypatch.setattr(torch.profiler, "record_function", fake_record_function)
    return names


def test_update_loop_names_each_mini_batch(monkeypatch):
    mini_batches = [TensorDict({}, batch_size=[]) for _ in range(3)]
    monkeypatch.setattr(tu, "make_iterator", lambda data, **kwargs: iter(mini_batches))
    names = _record_names(monkeypatch)

    engine = _engine()
    worker = _worker(engine)
    worker.train_batch = MagicMock(
        side_effect=lambda _data: (engine.optimizer.step(), {})[1]
    )

    data = TensorDict({}, batch_size=[3])
    tu.assign_non_tensor(data, num_mini_batch=3)

    assert TrainingWorker.train_mini_batch(worker, data) is None
    # Numbered from the start of the step: without the index every iteration looks alike.
    assert names == ["mini_batch0", "mini_batch1", "mini_batch2"]
    # The profiler is advanced once per mini-batch -- the unit a torch.profiler.schedule
    # sub-samples the update loop by.
    assert worker.profiler.step.call_count == len(mini_batches)


def test_forward_only_stage_is_not_a_profiler_step():
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, global_token_num=[4])
    worker = _worker(_engine())

    assert TrainingWorker.infer_batch(worker, data) is None
    worker.profiler.step.assert_not_called()


def test_update_loop_reports_actual_epoch_and_optimizer_step_counts(monkeypatch):
    mini_batches = [TensorDict({}, batch_size=[]) for _ in range(6)]
    monkeypatch.setattr(tu, "make_iterator", lambda data, **kwargs: iter(mini_batches))

    engine = _engine(is_mp_src_rank_with_outputs=lambda: True)
    worker = _worker(engine)
    outputs = iter(
        tu.get_tensordict(
            tensor_dict={},
            non_tensor_dict={"metrics": {"loss": float(index)}},
        )
        for index in range(6)
    )

    def train_batch(_data):
        engine.optimizer.step()
        return next(outputs)

    worker.train_batch = MagicMock(side_effect=train_batch)

    data = TensorDict({}, batch_size=[3])
    tu.assign_non_tensor(data, num_mini_batch=3, epochs=2)

    output = TrainingWorker.train_mini_batch(worker, data)
    metrics = tu.get(output, "metrics")
    assert metrics["ppo_epoch_passes_delta"].aggregate() == 2
    assert metrics["mini_batches_per_epoch"].aggregate() == 3
    assert metrics["optimizer_steps_delta"].aggregate() == 6
    assert worker.train_batch.call_count == 6


def test_update_loop_can_schedule_after_every_optimizer_step(monkeypatch):
    mini_batches = [TensorDict({}, batch_size=[]) for _ in range(2)]
    monkeypatch.setattr(tu, "make_iterator", lambda data, **kwargs: iter(mini_batches))

    engine = _engine()
    worker = _worker(engine)
    worker.optimizer_config.lr_scheduler_step_per_optimizer_step = True
    scheduler_flags = []

    def train_batch(mini_batch):
        scheduler_flags.append(tu.get(mini_batch, "update_lr_scheduler"))
        engine.optimizer.step()
        return {}

    worker.train_batch = train_batch
    data = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor(data, num_mini_batch=2, epochs=1)

    assert TrainingWorker.train_mini_batch(worker, data) is None
    assert scheduler_flags == [True, True]


def test_update_loop_rejects_non_positive_epoch_count():
    worker = _worker(_engine())
    data = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor(data, num_mini_batch=1, epochs=0)

    try:
        TrainingWorker.train_mini_batch(worker, data)
    except ValueError as exc:
        assert "epochs must be a positive integer" in str(exc)
    else:
        raise AssertionError("epochs=0 must fail closed")


def test_update_loop_rejects_truncated_iterator(monkeypatch):
    monkeypatch.setattr(
        tu,
        "make_iterator",
        lambda data, **kwargs: iter([TensorDict({}, batch_size=[])]),
    )

    engine = _engine()
    worker = _worker(engine)
    worker.train_batch = MagicMock(
        side_effect=lambda _data: (engine.optimizer.step(), {})[1]
    )
    data = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor(data, num_mini_batch=2, epochs=1)

    try:
        TrainingWorker.train_mini_batch(worker, data)
    except RuntimeError as exc:
        assert "unexpected number of train-batch calls" in str(exc)
    else:
        raise AssertionError("a truncated training iterator must fail closed")


def test_update_loop_rejects_skipped_optimizer_step(monkeypatch):
    monkeypatch.setattr(
        tu,
        "make_iterator",
        lambda data, **kwargs: iter([TensorDict({}, batch_size=[])]),
    )

    worker = _worker(_engine())
    # Simulate a completed forward/backward whose mixed-precision optimizer
    # update was skipped: no underlying optimizer.step() call reaches the hook.
    worker.train_batch = MagicMock(return_value={})
    data = TensorDict({}, batch_size=[1])
    tu.assign_non_tensor(data, num_mini_batch=1, epochs=1)

    try:
        TrainingWorker.train_mini_batch(worker, data)
    except RuntimeError as exc:
        assert "successful optimizer steps" in str(exc)
    else:
        raise AssertionError("a skipped optimizer update must fail closed")


def _run_full_batch_token_update(*, micro_partitions, include_padding):
    ratios = torch.tensor(
        [[0.1, 0.2], [1.0, 4.0], [5.0, 0.5], [2.0, 4.001], [0.199, 1.5]],
        dtype=torch.float64,
    )
    advantages = torch.tensor(
        [[1.0, -0.5], [0.25, 2.0], [-1.0, 0.75], [1.5, -2.0], [0.5, 1.25]],
        dtype=torch.float64,
    )
    features = torch.tensor(
        [[0.5, 1.0], [-0.75, 0.25], [1.25, -0.5], [0.8, 1.5], [-1.0, 0.4]],
        dtype=torch.float64,
    )
    old_log_probs = torch.full_like(ratios, -10.0)
    response_mask = torch.ones_like(ratios)
    if include_padding:
        padding_rows = 2
        ratios = torch.cat([ratios, torch.ones((padding_rows, 2), dtype=ratios.dtype)])
        advantages = torch.cat([advantages, torch.full((padding_rows, 2), 999.0, dtype=advantages.dtype)])
        features = torch.cat([features, torch.full((padding_rows, 2), 999.0, dtype=features.dtype)])
        old_log_probs = torch.cat([old_log_probs, torch.zeros((padding_rows, 2), dtype=old_log_probs.dtype)])
        response_mask = torch.cat([response_mask, torch.zeros((padding_rows, 2), dtype=response_mask.dtype)])

    data = TensorDict(
        {
            "log_ratio_offset": ratios.log(),
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "features": features,
            "response_mask": response_mask,
        },
        batch_size=[ratios.shape[0]],
    )
    tu.assign_non_tensor(
        data,
        mini_batch_size=ratios.shape[0],
        epochs=1,
        dataloader_kwargs={"shuffle": False},
    )

    parameter = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    observed_steps = []
    external_step_hook = optimizer.register_step_post_hook(
        lambda _optimizer, _args, _kwargs: observed_steps.append(1)
    )
    engine = _engine(optimizer=optimizer)
    worker = _worker(engine)
    config = SimpleNamespace(
        policy_loss={
            "rollout_correction": {
                "loss_type": "reinforce",
                "rollout_is": "token",
                "rollout_is_threshold": "0.2_4.0",
                "rollout_is_batch_normalize": False,
                "rollout_rs": None,
                "rollout_rs_threshold": None,
            }
        },
        global_batch_info={
            "dp_size": 1,
            "batch_num_tokens": int(response_mask.sum().item()),
            "global_batch_size": int(ratios.shape[0]),
        },
    )

    def train_batch(mini_batch):
        optimizer.zero_grad()
        cursor = 0
        for size in micro_partitions:
            micro_batch = mini_batch[cursor : cursor + size]
            cursor += size
            current_log_probs = (
                micro_batch["old_log_probs"]
                + micro_batch["log_ratio_offset"]
                + parameter * micro_batch["features"]
            )
            loss, _ = compute_policy_loss_bypass_mode(
                old_log_prob=micro_batch["old_log_probs"],
                log_prob=current_log_probs,
                advantages=micro_batch["advantages"],
                response_mask=micro_batch["response_mask"],
                loss_agg_mode="token-mean",
                config=config,
            )
            loss.backward()
        assert cursor == len(mini_batch)
        optimizer.step()
        return {}

    worker.train_batch = train_batch
    try:
        assert TrainingWorker.train_mini_batch(worker, data) is None
    finally:
        external_step_hook.remove()
    return parameter.detach(), len(observed_steps), ratios[:5], advantages[:5], features[:5]


def test_full_learner_batch_token_mean_is_partition_and_padding_invariant():
    whole, whole_steps, ratios, advantages, features = _run_full_batch_token_update(
        micro_partitions=(5,), include_padding=False
    )
    partitioned, partitioned_steps, _, _, _ = _run_full_batch_token_update(
        micro_partitions=(2, 1, 2), include_padding=False
    )
    padded, padded_steps, _, _, _ = _run_full_batch_token_update(
        micro_partitions=(2, 3, 2), include_padding=True
    )

    eligible = (ratios >= 0.2) & (ratios <= 4.0)
    expected_gradient = (
        -advantages * features * ratios * eligible.to(dtype=ratios.dtype)
    ).sum() / ratios.numel()
    expected_parameter = -0.1 * expected_gradient

    assert whole_steps == partitioned_steps == padded_steps == 1
    torch.testing.assert_close(whole, expected_parameter, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(partitioned, whole, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(padded, whole, rtol=0.0, atol=1e-12)
    assert int(eligible.sum().item()) == 6


def test_bypass_icepop_metrics_pool_exactly_across_valid_and_padding_microbatches():
    config = SimpleNamespace(
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        global_batch_info={
            "dp_size": 2,
            "batch_num_tokens": 6,
            "global_batch_size": 4,
        },
        policy_loss={
            "rollout_correction": {
                "loss_type": "reinforce",
                "rollout_is": "token",
                "rollout_is_threshold": "0.2_4.0",
                "rollout_is_batch_normalize": False,
                "rollout_rs": None,
                "rollout_rs_threshold": None,
            }
        },
    )
    rollout_log_prob = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]], dtype=torch.float64)
    current_log_prob = rollout_log_prob + torch.log(
        torch.tensor([[0.5, 1.0], [2.0, 8.0]], dtype=torch.float64)
    )
    response_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.float64)
    advantages = torch.ones_like(response_mask)

    _, full = compute_policy_loss_bypass_mode(
        old_log_prob=rollout_log_prob,
        log_prob=current_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=config,
    )
    _, first = compute_policy_loss_bypass_mode(
        old_log_prob=rollout_log_prob[:1],
        log_prob=current_log_prob[:1],
        advantages=advantages[:1],
        response_mask=response_mask[:1],
        loss_agg_mode="token-mean",
        config=config,
    )
    _, second = compute_policy_loss_bypass_mode(
        old_log_prob=rollout_log_prob[1:],
        log_prob=current_log_prob[1:],
        advantages=advantages[1:],
        response_mask=response_mask[1:],
        loss_agg_mode="token-mean",
        config=config,
    )
    _, padding = compute_policy_loss_bypass_mode(
        old_log_prob=torch.zeros((1, 2), dtype=torch.float64),
        log_prob=torch.zeros((1, 2), dtype=torch.float64, requires_grad=True),
        advantages=torch.zeros((1, 2), dtype=torch.float64),
        response_mask=torch.zeros((1, 2), dtype=torch.float64),
        loss_agg_mode="token-mean",
        config=config,
    )

    critical = {
        "rollout_corr/rollout_is_mean": AggregationType.MEAN,
        "rollout_corr/rollout_is_min": AggregationType.MIN,
        "rollout_corr/rollout_is_max": AggregationType.MAX,
        "rollout_corr/rollout_is_std": AggregationType.STD,
        "rollout_corr/rollout_is_eff_sample_size": AggregationType.EFFECTIVE_SAMPLE_SIZE,
        "rollout_corr/rollout_is_oob_ratio": AggregationType.MEAN,
    }
    for key, aggregation in critical.items():
        assert isinstance(full[key], Metric)
        assert full[key].aggregation == aggregation
        assert isinstance(first[key], Metric)
        assert isinstance(second[key], Metric)
        empty = first[key].init_list()
        assert Metric.aggregate_dp([first[key], second[key], empty]) == pytest.approx(
            full[key].aggregate(), rel=1e-12, abs=1e-12
        )
        assert key not in padding


def test_bypass_icepop_boundary_diagnostics_match_tolerant_partition():
    config = SimpleNamespace(
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        global_batch_info={
            "dp_size": 1,
            "batch_num_tokens": 6,
            "global_batch_size": 6,
        },
        policy_loss={
            "rollout_correction": {
                "loss_type": "reinforce",
                "rollout_is": "token",
                "rollout_is_threshold": "0.2_4.0",
                "rollout_is_batch_normalize": False,
                "rollout_rs": None,
                "rollout_rs_threshold": None,
            }
        },
    )
    dtype = torch.float64
    lower = torch.tensor(0.2, dtype=dtype)
    upper = torch.tensor(4.0, dtype=dtype)
    eps = torch.finfo(dtype).eps
    raw_weights = torch.stack(
        [
            lower - 32 * eps,
            torch.nextafter(lower, torch.tensor(float("-inf"), dtype=dtype)),
            lower,
            upper,
            torch.nextafter(upper, torch.tensor(float("inf"), dtype=dtype)),
            upper + 32 * eps * upper,
        ]
    ).unsqueeze(-1)
    rollout_log_prob = torch.zeros_like(raw_weights)

    _, metrics = compute_policy_loss_bypass_mode(
        old_log_prob=rollout_log_prob,
        log_prob=torch.log(raw_weights),
        advantages=torch.ones_like(raw_weights),
        response_mask=torch.ones_like(raw_weights),
        loss_agg_mode="token-mean",
        config=config,
    )

    low_fraction = metrics["rollout_corr/rollout_is_ratio_fraction_low"].aggregate()
    high_fraction = metrics["rollout_corr/rollout_is_ratio_fraction_high"].aggregate()
    oob_fraction = metrics["rollout_corr/rollout_is_oob_ratio"].aggregate()
    assert low_fraction == pytest.approx(1.0 / 6.0)
    assert high_fraction == pytest.approx(1.0 / 6.0)
    assert oob_fraction == pytest.approx(low_fraction + high_fraction)
    assert metrics["rollout_corr/rollout_is_seq_fraction_low"].aggregate() == pytest.approx(
        low_fraction
    )
    assert metrics["rollout_corr/rollout_is_seq_fraction_high"].aggregate() == pytest.approx(
        high_fraction
    )
