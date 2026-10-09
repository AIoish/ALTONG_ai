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
