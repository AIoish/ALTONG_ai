"""용도: 다섯 숫자 후보의 loss 범위와 입력 분리를 검증한다.
생성일: 2026-10-04
"""

import unittest
from pathlib import Path
import torch
from filtering_training.common.digit_scores import digit_loss, replace_prompt


requires_local_artifacts = unittest.skipUnless(
    Path("filtering_training/outputs/v3_reviewed_01/dataset.jsonl").is_file(),
    "local experiment data and adapters are not distributed through Git",
)


class DigitScoreTests(unittest.TestCase):
    def test_encoder_urgency_ignores_window_changes_and_variant_ids(self):
        import copy
        import json
        from filtering_training.modeling.encoder_score_experiment import input_text
        original = {'notification': {'id': 'notice_current', 'body': '작업실 가스 누출, 즉시 대피'},
                    'context': {'active_process': 'code.exe', 'window_title': 'review.md',
                                'recent_windows': [{'app_name': 'chrome.exe', 'window_title': 'Instagram'}]}}
        changed = copy.deepcopy(original)
        changed['notification']['id'] = 'notice_empty'
        changed['context']['window_title'] = ''
        changed['context']['recent_windows'] = []
        self.assertEqual(input_text(original, 'urgency'), input_text(changed, 'urgency'))
        self.assertNotEqual(input_text(original, 'relevance'), input_text(changed, 'relevance'))
        self.assertNotIn('id', json.loads(input_text(original, 'relevance'))['notification'])
        self.assertEqual(original['context']['window_title'], 'review.md')

    def test_encoder_metrics_distinguish_threshold_error_from_within_one(self):
        from types import SimpleNamespace
        from filtering_training.modeling.encoder_score_experiment import score
        samples = [SimpleNamespace(notification=SimpleNamespace(id=f'case_{i}'),
                                  label=SimpleNamespace(urgency_score=i, relevance_score=6-i)) for i in range(1, 6)]
        predictions = [{'urgency_score': i, 'relevance_score': 6-i} for i in range(1, 6)]
        perfect = score(samples, predictions)
        self.assertEqual(perfect['urgency_score']['macro_f1'], 1)
        self.assertEqual(perfect['relevance_score']['macro_f1'], 1)
        predictions[3]['urgency_score'] = 3
        actual = score(samples, predictions)
        self.assertEqual(actual['urgency_score']['within_one_accuracy'], 1)
        self.assertEqual(actual['policy_correct'], 4)
        self.assertEqual(actual['urgent_false_blocks'], 1)

    def test_cpu_completion_keeps_only_selected_adapter_and_digit_output(self):
        from filtering_training.modeling.benchmark_digit_cpu import completion_payload, parse_choice
        for task, ident in (("urgency", 0), ("relevance", 1)):
            request = completion_payload([10, 11], task)
            self.assertEqual(request["lora"], [{"id": ident, "scale": 1.0}])
            self.assertEqual(request["n_predict"], 1)
            self.assertEqual(request["grammar"], "root ::= [1-5]")
            self.assertFalse(request["cache_prompt"])
        self.assertEqual(parse_choice({"content": "4", "tokens": [19]}, [16,17,18,19,20]), 4)
        for invalid in ({"content": "4", "tokens": [18]}, {"content": "4", "tokens": [19], "truncated": True},
                        {"content": "reason: 4", "tokens": [19]}):
            with self.assertRaises(ValueError):
                parse_choice(invalid, [16,17,18,19,20])

    def test_laya_task_weighting_preserves_budget_and_scales_gradients(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import laya.train as sdk
        from filtering_training.modeling.laya_score_experiment import task_loss_weights, train_task_weighted
        counts = {"urgency": 4946, "relevance": 10173}
        share = 4936 / (4936 + 7610)
        weights = task_loss_weights(counts, share)
        self.assertAlmostEqual(sum(counts[t] * weights[t] for t in counts), sum(counts.values()))
        self.assertAlmostEqual(counts["urgency"] * weights["urgency"] / sum(counts.values()), share)
        parameter = torch.tensor(2.0, requires_grad=True)
        items = [{"_filtering_task": t} for t in ("urgency", "relevance")]
        config = SimpleNamespace(micro_batch=1, loss="soft-ce", epochs=1)
        seen = []
        def loop(model, tok, rows, config, device, max_len, head_max_len):
            for item in reversed(rows):
                sdk.encode_item(tok, item)
                parameter.grad = None
                value = sdk.soft_ce_loss()
                value.backward()
                seen.append((item["_filtering_task"], value.item(), parameter.grad.item()))
            return [1.0]
        with patch.object(sdk, "encode_item", return_value={}) as encoder, \
             patch.object(sdk, "soft_ce_loss", side_effect=lambda: parameter.square()) as loss, \
             patch.object(sdk, "train_model", side_effect=loop):
            self.assertEqual(train_task_weighted(None, None, items, config, "cpu", 1024, 256, weights), [1.0])
            self.assertIs(sdk.encode_item, encoder)
            self.assertIs(sdk.soft_ce_loss, loss)
            self.assertEqual(encoder.call_count, 2)
        for task, value, gradient in seen:
            self.assertAlmostEqual(value, 4 * weights[task], places=5)
            self.assertAlmostEqual(gradient, 4 * weights[task], places=5)
        for invalid in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                task_loss_weights(counts, invalid)
        config.micro_batch = 2
        with self.assertRaisesRegex(ValueError, "micro_batch=1"):
            train_task_weighted(None, None, items, config, "cpu", 1024, 256, weights)

    def test_laya_task_weighting_rejects_missing_losses_and_restores_sdk(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import laya.train as sdk
        from filtering_training.modeling.laya_score_experiment import train_task_weighted
        config = SimpleNamespace(micro_batch=1, loss="soft-ce", epochs=1)
        with patch.object(sdk, "train_model", return_value=[]) as trainer:
            encode, loss = sdk.encode_item, sdk.soft_ce_loss
            with self.assertRaisesRegex(ValueError, "exact training population"):
                train_task_weighted(None, None, [{"_filtering_task": "urgency"}], config,
                                    "cpu", 1024, 256, {"urgency": 1.0})
            self.assertIs(sdk.encode_item, encode)
            self.assertIs(sdk.soft_ce_loss, loss)

    def test_laya_task_weighting_runs_actual_sdk_loop(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import laya.train as sdk
        from filtering_training.modeling.laya_score_experiment import train_task_weighted
        model = torch.nn.Linear(1, 1)
        config = sdk.TrainConfig(epochs=1, micro_batch=1, grad_accum=2, loss="soft-ce",
                                amp=False, gradient_checkpointing=False, log_every=0)
        items = [{"_filtering_task": task, "q": {"t": "choice"}, "k": 5}
                 for task in ("urgency", "relevance")]
        batch = {"marker_mask": torch.ones(1), "target": torch.ones(1)}
        seen = []
        def encode(tok, item, *args):
            seen.append(item["_filtering_task"])
            return {}
        with patch.object(sdk, "encode_item", side_effect=encode), \
             patch.object(sdk, "collate_items", return_value=batch), \
             patch.object(sdk, "_forward", side_effect=lambda model, *args: model(torch.ones(1, 1))), \
             patch.object(sdk, "soft_ce_loss", side_effect=lambda logits, *args: logits.square().mean()):
            result = train_task_weighted(model, SimpleNamespace(pad_token_id=0), items, config,
                                        torch.device("cpu"), 1024, 256, {"urgency": 1.2, "relevance": .9})
        self.assertEqual(len(result), 1)
        self.assertEqual(sorted(seen), ["relevance", "urgency"])

    def test_laya_native_prepared_rejects_changed_hash_and_urgency_context(self):
        import json
        import tempfile
        from filtering_training.modeling import laya_score_experiment as experiment
        from filtering_training.preparation.prepare_score_experiment import write_json, write_lines, digest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {"format": "native_laya_choice_v2", "prompt_version": "v2",
                        "protected_hashes": {}, "files": {}, "counts": {}}
            for task in ("urgency", "relevance"):
                (root / task).mkdir()
                state = {"notification": {"body": "example"}}
                if task == "relevance":
                    state["context"] = {"window_title": "example task"}
                row = {"state": state, "questions": {task: experiment.question(task, "", prompt_version="v2")},
                       "expected": {task: "2"}}
                path = root / task / "train.jsonl"
                write_lines(path, [row])
                manifest["files"][task + "/train.jsonl"] = {"count": 1, "sha256": digest(path)}
                manifest["counts"][task] = {"total": 1}
            write_json(root / "manifest.json", manifest)
            sources, _ = experiment.load_native_prepared(root)
            path = root / "urgency/train.jsonl"
            row = sources["urgency"][0]
            row["expected"]["urgency"] = "3"
            write_lines(path, [row])
            with self.assertRaisesRegex(ValueError, "protected input changed"):
                experiment.load_native_prepared(root)
            row["state"]["context"] = {"window_title": "must not enter urgency"}
            write_lines(path, [row])
            manifest["files"]["urgency/train.jsonl"]["sha256"] = digest(path)
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "context separation"):
                experiment.load_native_prepared(root)

    def test_laya_fit_diagnosis_uses_only_actual_two_phase_training_rows(self):
        import hashlib
        import json
        import tempfile
        from unittest.mock import patch
        from filtering_training.checks.test_score_separation import sample
        from filtering_training.common.score_tasks import sft_record
        from filtering_training.modeling import laya_score_experiment as experiment
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = {}
            for task in ("urgency", "relevance"):
                sources[task] = [sft_record(sample(urgency=i, relevance=i), task)
                                 for i in range(1, 6) for _ in range(2)]
                (root / task).mkdir()
                (root / task / "train.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in sources[task]), encoding="utf-8")
            phases = [{"source_indices": {"urgency": [0, 2, 4, 6, 8], "relevance": [0, 2, 4, 6]}, "groups": ["A"]},
                      {"source_indices": {"urgency": [1, 3, 5, 7, 9], "relevance": [3, 5, 7, 8, 9]}, "groups": ["B"]}]
            parent = {"status": "completed", "selected_policy": "exclude",
                      "conflicting_relevance_rows": [{"source_index": 1}],
                      "proposed_alternatives": {"exclude": {"phases": phases}}}
            (root / "run_config.json").write_text(json.dumps(parent), encoding="utf-8")
            view = hashlib.sha256()
            for phase in phases:
                for task in sources:
                    for index in phase["source_indices"][task]:
                        row = experiment.training_row(sources[task][index], task, "v2")
                        view.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            stage = root / "phase100"
            stage.mkdir()
            run = {"excluded_relevance_rows": 1, "prompt_version": "v2",
                   "train_counts": {"urgency": 10, "relevance": 9}, "training_view_sha256": view.hexdigest()}
            (stage / "run_config.json").write_text(json.dumps(run), encoding="utf-8")
            with patch.object(experiment, "PREPARED", root), \
                 patch.object(experiment, "preflight", return_value=(None, None, None,
                     {"training_status": "project-fine-tuned Laya", "preserved_hashes": {}})), \
                 patch("builtins.print"):
                result = experiment.diagnose_train(stage / "fit.json", stage / "checkpoint",
                                                   root / "unused.json", "cpu", dry_run=True)
                self.assertEqual(result["sampling"]["relevance"]["population_by_score"][1], 1)
                self.assertEqual(sum(result["sampling"]["relevance"]["population_by_score"].values()), 9)
                self.assertEqual(result["training_view_sha256"], view.hexdigest())

    def test_laya_half_plan_rejects_excluded_rows_overlap_and_missing_sources(self):
        from copy import deepcopy
        from filtering_training.modeling.laya_score_experiment import validate_half_indices
        phases = [{"source_indices": {"urgency": [0, 1, 2], "relevance": [0, 1]}, "groups": ["A"]},
                  {"source_indices": {"urgency": [3, 4, 5], "relevance": [2, 3, 4]}, "groups": ["B"]}]
        validate_half_indices(phases, {"urgency": 6, "relevance": 6}, {5})
        for mutation in ("excluded", "overlap", "missing", "group"):
            invalid = deepcopy(phases)
            if mutation == "excluded":
                invalid[1]["source_indices"]["relevance"].append(5)
            elif mutation == "overlap":
                invalid[1]["source_indices"]["urgency"].append(0)
            elif mutation == "missing":
                invalid[0]["source_indices"]["urgency"].pop()
            else:
                invalid[1]["groups"] = ["A"]
            with self.assertRaises(ValueError):
                validate_half_indices(invalid, {"urgency": 6, "relevance": 6}, {5})

    def test_laya_v2_training_and_inference_use_identical_questions(self):
        from filtering_training.checks.test_score_separation import sample
        from filtering_training.common.score_tasks import sft_record
        from filtering_training.modeling.laya_score_experiment import request, training_row
        source = sample(urgency=5, relevance=4)
        for task in ("urgency", "relevance"):
            state, questions = request(source, task, prompt_version="v2")
            train = training_row(sft_record(source, task), task, prompt_version="v2")
            self.assertEqual((train["state"], train["questions"]), (state, questions))
            self.assertEqual(train["expected"], {task: str(getattr(source.label, task + "_score"))})
            self.assertNotIn("id", state["notification"])
            if task == "urgency":
                self.assertNotIn("context", state)

    @unittest.skipUnless(Path("filtering_training/outputs/models/laya-multilingual/tokenizer/tokenizer.json").is_file(),
                         "pinned multilingual tokenizer is a local artifact")
    def test_laya_v2_question_and_all_options_fit_without_clipping(self):
        from transformers import AutoTokenizer
        from filtering_training.modeling.laya_score_experiment import question, question_budget
        tok = AutoTokenizer.from_pretrained("filtering_training/outputs/models/laya-multilingual/tokenizer",
                                           local_files_only=True)
        for task in ("urgency", "relevance"):
            for reverse in (False, True):
                q = question(task, "", reverse=reverse, prompt_version="v2")
                budget = question_budget(tok, task, {task: q})
                self.assertFalse(budget["clipped"])
                self.assertLessEqual(budget["head_tokens"], 256)
                self.assertEqual(set(q["criteria"]), {"1", "2", "3", "4", "5"})

    def test_laya_fit_diagnosis_deduplicates_and_checks_actual_training_hash(self):
        import hashlib
        import json
        import tempfile
        from unittest.mock import patch
        from filtering_training.checks.test_score_separation import sample
        from filtering_training.common.score_tasks import sft_record
        from filtering_training.modeling import laya_score_experiment as experiment
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            converted = hashlib.sha256()
            for task in ("urgency", "relevance"):
                records = [sft_record(sample(urgency=i, relevance=i), task)
                           for i in range(1, 6) for _ in range(2)]
                target = root / task
                target.mkdir()
                (target / "train.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in records), encoding="utf-8")
                for record in records:
                    converted.update((json.dumps(experiment.training_row(record, task),
                                                 ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            run_file = root / "run_config.json"
            run = {"train_counts": {"urgency": 10, "relevance": 10},
                   "training_view_sha256": converted.hexdigest()}
            run_file.write_text(json.dumps(run), encoding="utf-8")
            plan = {"training_status": "project-fine-tuned Laya"}
            with patch.object(experiment, "PREPARED", root), \
                 patch.object(experiment, "preflight", return_value=(None, None, None, plan)), \
                 patch("builtins.print"):
                output = root / "diagnostic.json"
                result = experiment.diagnose_train(output, root / "checkpoint", root / "unused.json",
                                                   "cpu", per_score=2, dry_run=True)
                self.assertEqual(result["evaluated_task_rows"], 10)
                self.assertEqual(result["sampling"]["urgency"]["selected_by_score"],
                                 {i: 1 for i in range(1, 6)})
                self.assertFalse(output.exists())
                run["training_view_sha256"] = "different"
                run_file.write_text(json.dumps(run), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "actual completed training run"):
                    experiment.diagnose_train(output, root / "checkpoint", root / "unused.json",
                                              "cpu", dry_run=True)

    def test_laya_training_conversion_preserves_labels_without_context_leakage(self):
        from filtering_training.checks.test_score_separation import sample
        from filtering_training.common.score_tasks import sft_record
        from filtering_training.modeling.laya_score_experiment import training_row
        source = sample(urgency=5, relevance=1)
        urgency = training_row(sft_record(source, "urgency"), "urgency")
        relevance = training_row(sft_record(source, "relevance"), "relevance")
        self.assertEqual(urgency["expected"], {"urgency": "5"})
        self.assertEqual(relevance["expected"], {"relevance": "1"})
        self.assertNotIn("context", urgency["state"])
        self.assertNotIn("id", relevance["state"]["notification"])
        self.assertEqual(relevance["state"]["context"]["window_title"], "secret-context")
        self.assertNotIn("gold", relevance["state"])
        self.assertNotIn("label", relevance["state"])

    def test_laya_training_rejects_non_integer_target(self):
        import json
        from filtering_training.checks.test_score_separation import sample
        from filtering_training.common.score_tasks import sft_record
        from filtering_training.modeling.laya_score_experiment import training_row
        record = sft_record(sample(), "urgency")
        record["messages"][-1]["content"] = json.dumps({"urgency_score": True})
        with self.assertRaisesRegex(ValueError, "original label range"):
            training_row(record, "urgency")

    def test_laya_urgency_input_never_contains_work_context_or_variant_id(self):
        from filtering_training.checks.test_score_separation import sample
        from filtering_training.modeling.laya_score_experiment import request
        state, questions = request(sample(), "urgency")
        self.assertNotIn("context", state)
        self.assertNotIn("id", state["notification"])
        self.assertNotIn("secret-context", str(state))
        self.assertEqual(questions["urgency"]["type"], "choice")

    def test_laya_reversing_options_preserves_score_meaning_and_context(self):
        from filtering_training.checks.test_score_separation import sample
        from filtering_training.modeling.laya_score_experiment import request
        normal_state, normal = request(sample(), "relevance")
        reversed_state, reversed_questions = request(sample(), "relevance", reverse=True)
        self.assertEqual(normal_state, reversed_state)
        self.assertEqual(normal_state["context"]["window_title"], "secret-context")
        self.assertNotIn("id", normal_state["notification"])
        self.assertEqual(normal["relevance"]["criteria"], reversed_questions["relevance"]["criteria"])
        self.assertEqual(list(reversed_questions["relevance"]["criteria"]), ["5", "4", "3", "2", "1"])

    def test_epoch_schedule_counts_optimizer_steps_and_partial_tail(self):
        from filtering_training.modeling.digit_score_experiment import training_schedule
        for count,steps,warmup in ((4936,617,31),(11771,1472,74)):
            plan=training_schedule(count,8,1,warmup=.05)
            self.assertEqual(plan["expected_optimizer_steps"],steps)
            self.assertEqual(plan["expected_examples"],count)
            self.assertEqual(plan["warmup_steps"],warmup)
            self.assertEqual(plan["trainer_max_steps"],-1)
        for kwargs in ({"accumulation":8},{"accumulation":0},{"epochs":1,"max_steps":10},{"epochs":0},{"epochs":1,"warmup":1}):
            with self.assertRaises(ValueError):
                training_schedule(11,**kwargs)

    def test_accumulation_matches_full_batch_gradient_including_partial_tail(self):
        import tempfile
        from types import SimpleNamespace
        from transformers import Trainer, TrainingArguments
        from torch.utils.data import SequentialSampler
        from filtering_training.modeling.digit_score_experiment import digit_trainer_class
        labels=[0,1,1,2,3,4,4,4,0,2,4]
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.zeros(5))
            def forward(self,input_ids,**kwargs):
                return SimpleNamespace(logits=self.weight.reshape(1,1,5).expand(len(input_ids),1,5))
        class SequentialTrainer(digit_trainer_class(Trainer,list(range(5)))):
            def _get_train_sampler(self,train_dataset=None):
                return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
        model=Model()
        with tempfile.TemporaryDirectory() as temp:
            args=TrainingArguments(output_dir=temp,use_cpu=True,per_device_train_batch_size=1,
                gradient_accumulation_steps=8,num_train_epochs=1,learning_rate=.1,optim="sgd",
                lr_scheduler_type="constant",max_grad_norm=0,report_to="none",save_strategy="no",
                logging_strategy="no",disable_tqdm=True,remove_unused_columns=False)
            trainer=SequentialTrainer(model=model,args=args,train_dataset=[{"input_ids":[0],"labels":v} for v in labels])
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            trainer.train()
            self.assertEqual(trainer.training_examples_seen,11)
            self.assertEqual(trainer.state.global_step,2)
        reference=torch.zeros(5,requires_grad=True)
        optimizer=torch.optim.SGD([reference],lr=.1)
        for chunk in (labels[:8],labels[8:]):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(reference.expand(len(chunk),5),torch.tensor(chunk)).backward()
            optimizer.step()
        torch.testing.assert_close(model.weight,reference,atol=1e-6,rtol=1e-6)

    @requires_local_artifacts
    def test_latest_adapter_can_reuse_identical_prepared_data_without_rewriting_manifest(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from filtering_training.modeling import digit_score_experiment as module
        root=Path("filtering_training/outputs/v5_training_pool_01/prepared_continual")
        parent=Path("filtering_training/outputs/runs/2026-10-05-qwen06-continual-01")
        with tempfile.TemporaryDirectory() as temp:
            for task,count,steps in (("urgency",4936,617),("relevance",11771,1472)):
                with patch.object(module,"load_model",side_effect=AssertionError("no model loading")):
                    plan=module.train(Path(temp)/task,task,"natural",False,root,continue_from=parent/task,
                                      dry_run=True,accumulation=8,epochs=1,warmup=.05)
                self.assertEqual(plan["max_steps"],steps)
                self.assertEqual(plan["schedule"]["expected_examples"],count)
                self.assertFalse((Path(temp)/task).exists())

    @requires_local_artifacts
    def test_continuation_dry_run_never_loads_model_or_creates_output(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from filtering_training.modeling import digit_score_experiment as module
        root=Path("filtering_training/outputs/v5_training_pool_01/prepared_continual")
        parents=Path("filtering_training/outputs/runs/2026-10-05-qwen06-accumulated-01")
        with tempfile.TemporaryDirectory() as temp:
            for task,parent,count in (("urgency","urgency",4936),("relevance","relevance",11771)):
                output=Path(temp)/task
                with patch.object(module,"load_model",side_effect=AssertionError("model loading forbidden")):
                    plan=module.train(output,task,"natural",False,root,continue_from=parents/parent,dry_run=True)
                self.assertFalse(output.exists())
                self.assertFalse(plan["training_executed"])
                self.assertEqual(plan["train_rows"],count)
                self.assertEqual(plan["max_steps"],count)
                self.assertEqual(plan["learning_rate"],.00005)
                self.assertEqual(plan["validation_rows"],74)

    @requires_local_artifacts
    def test_continuation_rejects_wrong_task_and_mutated_parent(self):
        import tempfile
        import shutil
        from pathlib import Path
        from safetensors.torch import save_file
        from unittest.mock import patch
        from filtering_training.modeling import digit_score_experiment as module
        parents=Path("filtering_training/outputs/runs/2026-10-04-qwen06-pool-02")
        with self.assertRaisesRegex(ValueError,"matching score adapter"):
            module.continuation_snapshot(parents/"urgency","relevance")
        original_digest=module.digest
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp)/"parent"
            (parent/"adapter").mkdir(parents=True)
            shutil.copyfile(parents/"urgency/run_config.json",parent/"run_config.json")
            shutil.copyfile(parents/"urgency/adapter/adapter_config.json",parent/"adapter/adapter_config.json")
            # A temporary changed LoRA file keeps this rejection test independent of deleted historical weights.
            save_file({"test.lora_A.weight":torch.zeros(1,1)},parent/"adapter/adapter_model.safetensors")
            with patch.object(module,"digest",side_effect=lambda p:"mutated" if p.name=="adapter_model.safetensors" else original_digest(p)):
                with self.assertRaisesRegex(ValueError,"planned continuation adapter changed"):
                    module.preflight_training(Path(temp)/"unused","urgency",
                        Path("filtering_training/outputs/v5_training_pool_01/prepared_continual"),parent)
            self.assertFalse((Path(temp)/"unused").exists())

    def test_continuation_restores_trainable_adapter_without_initializing_new_weights(self):
        from pathlib import Path
        from unittest.mock import patch
        from filtering_training.modeling.digit_score_experiment import attach_training_adapter
        model=object(); result=object()
        with patch("peft.PeftModel.from_pretrained",return_value=result) as restore,patch("peft.get_peft_model") as initialize:
            self.assertIs(attach_training_adapter(model,{"run":"existing"}),result)
            restore.assert_called_once_with(model,Path("existing/adapter"),is_trainable=True)
            initialize.assert_not_called()

    def test_continuation_runner_plans_baseline_training_and_comparison_without_execution(self):
        import tempfile
        from pathlib import Path
        from filtering_training.modeling.run_score_pilot import build_plan
        parent=Path("filtering_training/outputs/runs/2026-10-04-qwen06-pool-02")
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"not_created"
            plan=build_plan("continual",Path("filtering_training/outputs/v5_training_pool_01/prepared_continual"),output,parent)
            self.assertEqual(len(plan),5)
            self.assertEqual(plan[0][2][0],"evaluate")
            for _,_,args in plan[1:3]:
                self.assertEqual(args[0],"train")
                self.assertIn("--continue-from",args)
                self.assertIn("--learning-rate",args)
            self.assertEqual(plan[3][2][0],"evaluate")
            self.assertIn("continual",plan[4][2])
            self.assertFalse(output.exists())

    @requires_local_artifacts
    def test_expanded_validation_keeps_legacy_and_only_approved_rows(self):
        from filtering_training.modeling.digit_score_experiment import load_validation_for_evaluation, VALIDATION_POOL
        from filtering_training.common.score_tasks import unique_urgency_samples
        old,old_groups,_=load_validation_for_evaluation()
        expanded,groups,subset=load_validation_for_evaluation(VALIDATION_POOL)
        self.assertEqual(len(old),24)
        self.assertEqual([s.model_dump(mode="json") for s in expanded[:24]],[s.model_dump(mode="json") for s in old])
        self.assertEqual(len(expanded),74)
        self.assertEqual(len(unique_urgency_samples(expanded)),31)
        self.assertEqual(groups["legacy_24"],old_groups["legacy_24"])
        self.assertEqual(len(groups["approved_expansion"]),50)
        self.assertEqual(subset["excluded_unreviewed_rows"],330)

    @requires_local_artifacts
    def test_expanded_validation_rejects_unreviewed_rows_even_with_updated_hash(self):
        import json, tempfile
        from pathlib import Path
        from filtering_training.modeling.digit_score_experiment import load_validation_for_evaluation, VALIDATION_POOL
        from filtering_training.preparation.prepare_score_experiment import digest,write_json,write_lines
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ("manifest.json","approval.json","dataset.jsonl"):
                (root/name).write_bytes((VALIDATION_POOL/name).read_bytes())
            manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
            rows=[json.loads(line) for line in (root/"dataset.jsonl").read_text(encoding="utf-8").splitlines()]
            allowed=set(manifest["evaluation_set"]["approved_ids"])
            unreviewed=next(json.loads(line) for line in (VALIDATION_POOL/"candidates.jsonl").read_text(encoding="utf-8").splitlines()
                            if json.loads(line)["notification"]["id"] not in allowed)
            write_lines(root/"dataset.jsonl",rows+[unreviewed])
            manifest["evaluation_set"]["dataset_sha256"]=digest(root/"dataset.jsonl")
            manifest["evaluation_set"]["approved_ids"].append(unreviewed["notification"]["id"])
            write_json(root/"manifest.json",manifest)
            with self.assertRaisesRegex(ValueError,"unreviewed validation"):
                load_validation_for_evaluation(root)

    def test_training_subset_rejects_invalid_indices_and_preserves_selection(self):
        from filtering_training.modeling.digit_score_experiment import select_indices
        self.assertEqual(select_indices([1,4,5],"natural",[2,0]),[2,0])
        for subset in ([0,0],[-1],[3],[],[True]):
            with self.assertRaises(ValueError):
                select_indices([1,4,5],"natural",subset)
        with self.assertRaises(ValueError):
            select_indices([1,4,5],"balanced",[0])

    def test_unified_runner_preserves_experiment_jobs_without_creating_output(self):
        import tempfile
        from pathlib import Path
        from filtering_training.modeling.run_score_pilot import build_plan, PREPARED, BASELINE
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"not_created"
            for experiment,count in (("pilot",8),("controls",12),("digit",6),("supplement",6),("threshold",3),("epochs3",4),("fit",2),("pool",3)):
                plan = build_plan(experiment,PREPARED,output,BASELINE)
                self.assertEqual(len(plan),count)
                self.assertTrue(all(log.is_relative_to(output) for log,_,_ in plan))
                self.assertFalse(output.exists())
            threshold = build_plan("threshold",PREPARED,output,BASELINE)
            self.assertIn("--threshold-weight",threshold[0][2])
            self.assertIn(str(BASELINE/"relevance"),threshold[1][2])
            supplement = build_plan("supplement",PREPARED,output,BASELINE)
            for index,steps in ((0,"58"),(1,"174"),(3,"58"),(4,"174")):
                arguments = supplement[index][2]
                self.assertEqual(arguments[arguments.index("--max-steps")+1],steps)
            epochs = build_plan("epochs3",PREPARED,output,BASELINE)
            self.assertEqual(epochs[0][2][epochs[0][2].index("--max-steps")+1],"174")
            self.assertEqual(epochs[0][2][epochs[0][2].index("--threshold-weight")+1],"0")
            pool = build_plan("pool",PREPARED,output,BASELINE)
            for _,_,arguments in pool[:2]:
                self.assertEqual(arguments[arguments.index("--mode")+1],"natural")
                self.assertNotIn("--max-steps",arguments)
            self.assertNotIn("--relevance-run",pool[2][2])

    def test_threshold_targets_and_zero_weight_preserve_original_loss(self):
        import math
        logits = torch.zeros(1,1,10,requires_grad=True)
        ids = [1,2,3,4,5]
        for label,group_size in ((2,3),(3,2),(4,2)):
            labels = torch.tensor([label])
            original = digit_loss(logits,labels,ids)
            self.assertAlmostEqual(original.item(),math.log(5),places=5)
            combined = digit_loss(logits,labels,ids,1.0)
            self.assertAlmostEqual(combined.item(),math.log(5)+math.log(5/group_size),places=5)

    def test_threshold_auxiliary_pushes_mass_toward_urgent_group(self):
        ids = [1,2,3,4,5]
        logits = torch.zeros(1,1,10,requires_grad=True)
        base = digit_loss(logits,torch.tensor([4]),ids)
        combined = digit_loss(logits,torch.tensor([4]),ids,1.0)
        (combined-base).backward()
        self.assertTrue(bool((logits.grad[0,0,ids[:3]] > 0).all()))
        self.assertTrue(bool((logits.grad[0,0,ids[3:]] < 0).all()))

    def test_gradient_only_reaches_last_position_and_five_candidates(self):
        logits = torch.zeros(1,3,12,requires_grad=True)
        loss = digit_loss(logits,torch.tensor([3]),[1,3,5,7,9])
        self.assertAlmostEqual(loss.item(),1.6094379,places=5)
        loss.backward()
        self.assertEqual(logits.grad[0,:2].count_nonzero().item(),0)
        self.assertEqual(logits.grad[0,2].count_nonzero().item(),5)
        self.assertLess(logits.grad[0,2,7].item(),0)

    def test_reject_out_of_range_labels(self):
        with self.assertRaises(ValueError):
            digit_loss(torch.zeros(1,1,10),torch.tensor([5]),[1,2,3,4,5])

    def test_replaces_only_system_and_drops_completion(self):
        msgs = [{"role":"system","content":"old"},{"role":"user","content":"notification only"},
                {"role":"assistant","content":"gold must not enter prompt"}]
        result = replace_prompt(msgs,"urgency")
        self.assertEqual(result[1],msgs[1])
        self.assertEqual(len(result),2)
        self.assertNotEqual(result[0],msgs[0])

    def test_relevance_input_excludes_variant_id_but_preserves_task_context(self):
        import json
        payload={"notification":{"id":"sample_exact","body":"reference notice"},"context":{"window_title":"current task"}}
        messages=[{"role":"system","content":"old"},{"role":"user","content":json.dumps(payload)}]
        result=replace_prompt(messages,"relevance")
        actual=json.loads(result[1]["content"])
        self.assertNotIn("id",actual["notification"])
        self.assertEqual(actual["notification"]["body"],payload["notification"]["body"])
        self.assertEqual(actual["context"],payload["context"])
        self.assertEqual(json.loads(messages[1]["content"]),payload)


if __name__ == "__main__":
    unittest.main()
