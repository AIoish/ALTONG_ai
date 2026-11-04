"""용도: 기존 KLUE 학습·안전 보강 조건으로 긴급도 5단계와 관련도 3단계를 공동 학습한다.
생성일: 2026-11-03
"""

import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import random
import statistics
import time
import unicodedata

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_data import DEFAULT_DATASET, load_evaluation_inputs, load_windows_prepared
from filtering_training.common.encoder_input import (
    INPUT_SCHEMA, LEGACY_INPUT_SCHEMA, RECENT_WINDOWS_POLICY,
    encode_checked, input_text, legacy_input_text, windows_only_state,
)
from filtering_training.common.validation_data import evaluation_sets, validation

BASE = ROOT / 'filtering_training/outputs/models/klue-roberta-base'

OLD_BASE = ROOT/'filtering_training/outputs/runs/2026-10-10-klue-base-02'
OLD_SAFETY = ROOT/'filtering_training/outputs/runs/2026-10-10-base-safety-continue-01'
SAFETY = ROOT/'filtering_training/outputs/v10_safety_supplement_01'
TASKS = ('urgency', 'relevance')
RELEVANCE_MAP = {1:1, 2:2, 3:2, 4:3, 5:3}


def map_relevance(value):
    if type(value) is not int or value not in RELEVANCE_MAP:
        raise ValueError('Expected original integer relevance in 1..5')
    return RELEVANCE_MAP[value]


def passed(urgency, relevance):
    if type(urgency) is not int or not 1 <= urgency <= 5:
        raise ValueError('Urgency outside 1..5')
    if type(relevance) is not int or not 1 <= relevance <= 3:
        raise ValueError('Relevance outside 1..3')
    return urgency >= 4 or relevance == 3


def body_key(state):
    return ''.join(unicodedata.normalize('NFKC', state['notification'].get('body','')).split()).casefold()


def evaluate(rows, predictions):
    from sklearn.metrics import confusion_matrix, f1_score
    if len(rows) != len(predictions) or not rows:
        raise ValueError('Nonempty matching evaluation rows required')
    result = {'rows':len(rows)}
    ids = [r['notification']['id'] for r in rows]
    gold_u = [r['label']['urgency_score'] for r in rows]
    gold_r = [map_relevance(r['label']['relevance_score']) for r in rows]
    pred_u = [p['urgency_score'] for p in predictions]
    pred_r = [p['relevance_score'] for p in predictions]
    expected = [passed(u,r) for u,r in zip(gold_u,gold_r)]
    actual = [passed(u,r) for u,r in zip(pred_u,pred_r)]
    for task,gold,pred,classes in [('urgency',gold_u,pred_u,5),('relevance',gold_r,pred_r,3)]:
        result[task] = {
            'classes':classes,'exact_correct':sum(g==p for g,p in zip(gold,pred)),
            'exact_accuracy':sum(g==p for g,p in zip(gold,pred))/len(rows),
            'macro_f1':float(f1_score(gold,pred,labels=list(range(1,classes+1)),average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(gold,pred,labels=list(range(1,classes+1))).tolist(),
        }
    result.update(
        policy_correct=sum(g==p for g,p in zip(expected,actual)),
        policy_miss_ids=[ids[i] for i in range(len(rows)) if expected[i] and not actual[i]],
        policy_false_pass_ids=[ids[i] for i in range(len(rows)) if not expected[i] and actual[i]],
        urgent_count=sum(u>=4 for u in gold_u),
        urgency_threshold_miss_ids=[ids[i] for i in range(len(rows)) if gold_u[i]>=4 and pred_u[i]<4],
        urgent_policy_miss_ids=[ids[i] for i in range(len(rows)) if gold_u[i]>=4 and not actual[i]],
        relevance_positive_count=sum(r==3 for r in gold_r),
        relevance_pass_correct=sum((g==3)==(p==3) for g,p in zip(gold_r,pred_r)),
        relevance_miss_ids=[ids[i] for i in range(len(rows)) if gold_r[i]==3 and pred_r[i]!=3],
        relevance_false_pass_ids=[ids[i] for i in range(len(rows)) if gold_r[i]!=3 and pred_r[i]==3],
    )
    return result


def initial_rank(result):
    return (result['policy_correct'], -len(result['urgent_policy_miss_ids']),
            result['urgency']['exact_accuracy']+result['relevance']['exact_accuracy'])


def safety_rank(original, development):
    eligible = original['policy_correct'] >= 65 and len(original['urgent_policy_miss_ids']) <= 2
    return (int(eligible), -len(development['urgent_policy_miss_ids']),
            -len(original['urgent_policy_miss_ids']), -len(development['policy_false_pass_ids']),
            original['policy_correct'], development['policy_correct'])


def build_model(base=BASE, pretrained=True, attention='eager'):
    import torch
    from transformers import AutoConfig, AutoModel

    class JointEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            config = AutoConfig.from_pretrained(str(base),local_files_only=True)
            if config.model_type != 'roberta':
                raise ValueError('This controlled experiment expects KLUE RoBERTa')
            if pretrained:
                self.encoder = AutoModel.from_pretrained(str(base),local_files_only=True,
                                                        add_pooling_layer=False,attn_implementation=attention)
            else:
                self.encoder = AutoModel.from_config(config,add_pooling_layer=False,attn_implementation=attention)
            self.dropout = torch.nn.Dropout(.1)
            # Draw the original two five-class heads first to preserve the initial urgency
            # head and the post-initialization RNG state of the old seed-42 experiment.
            self.heads = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size,5) for _ in TASKS])
            old = self.heads[1]
            with torch.random.fork_rng(devices=[]):
                replacement = torch.nn.Linear(config.hidden_size,3)
            with torch.no_grad():
                replacement.weight.copy_(old.weight[:3])
                replacement.bias.copy_(old.bias[:3])
            self.heads[1] = replacement

        def forward(self,input_ids,attention_mask):
            hidden = self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0]
            hidden = self.dropout(hidden)
            return self.heads[0](hidden),self.heads[1](hidden)

    return JointEncoder()


def task_loss(outputs,tasks,labels,weights,denominator):
    import torch
    # Original stacked heads receive zero (not None) gradients for absent tasks;
    # preserve AdamW decay on both heads even in a single-task effective batch.
    loss = sum(logits.sum()*0 for logits in outputs)
    for task,logits in enumerate(outputs):
        selected = tasks == task
        if selected.any():
            loss = loss + torch.nn.functional.cross_entropy(logits[selected].float(),labels[selected],reduction='sum')*weights[task]/denominator
    return loss


def predictions(model,tokenizer,encoded):
    import torch
    model.eval()
    all_u,all_r = [],[]
    with torch.inference_mode():
        for start in range(0,len(encoded),8):
            batch = tokenizer.pad(encoded[start:start+8],padding=True,return_tensors='pt').to('cuda')
            urgency,relevance = model(**batch)
            all_u.extend((urgency.argmax(-1)+1).cpu().tolist())
            all_r.extend((relevance.argmax(-1)+1).cpu().tolist())
    return [{'urgency_score':all_u[2*i],'relevance_score':all_r[2*i+1]} for i in range(len(encoded)//2)]


def baseline(rows,tokenizer,parent):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads=4
    options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    options.add_session_config_entry('session.inter_op.allow_spinning','0')
    session=ort.InferenceSession(str(parent/'model-fp32.onnx'),options,providers=['CPUExecutionProvider'])
    result=[]
    for row in rows:
        values=[]
        for head,task in enumerate(TASKS):
            encoded=tokenizer(legacy_input_text(row,task),return_tensors='np',return_token_type_ids=False,truncation=False)
            if encoded['input_ids'].shape[1]>512:
                raise ValueError('Baseline exceeds token limit')
            logits=session.run(None,{k:np.asarray(v,dtype=np.int64) for k,v in encoded.items()})[0]
            values.append(int(logits[0,head].argmax())+1)
        result.append({'urgency_score':values[0],'relevance_score':map_relevance(values[1]),
                       'original_relevance5':values[1]})
    return result


def train_stage(folder,model,tokenizer,records,evals,eval_encoded,settings,selection):
    import torch
    import numpy as np
    from transformers import get_linear_schedule_with_warmup
    folder.mkdir()
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    # Stage one was seeded before model initialization; keep that RNG state.
    if selection=='safety':
        model.encoder.set_attn_implementation('sdpa')
    model.encoder.gradient_checkpointing_enable()
    tokenizer.save_pretrained(folder/'tokenizer')
    model.encoder.config.save_pretrained(folder/'encoder')
    counts=[sum(t==j for _,t,_ in records) for j in range(2)]
    weights=torch.tensor([len(records)/(2*c) for c in counts],device='cuda')
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':settings['encoder_lr']},
                                 {'params':model.heads.parameters(),'lr':settings['head_lr']}],weight_decay=.01,foreach=False)
    updates=math.ceil(len(records)/16)*settings['epochs_max']
    scheduler=get_linear_schedule_with_warmup(optimizer,int(updates*settings['warmup_ratio']),updates)
    scaler=torch.amp.GradScaler('cuda')
    config=dict(settings,status='training',created_date=created_date(),input_schema=INPUT_SCHEMA,output_classes={'urgency':5,'relevance':3},
                train_counts=dict(zip(TASKS,counts)),task_weights=dict(zip(TASKS,weights.cpu().tolist())),
                effective_batch=16,micro_batch=4,weight_decay=.01,maximum_tokens=512,
                checkpoint_selection=selection,optimizer_resumed=False)
    write(folder/'run_config.json',config)
    best,stale,history,successful=None,0,[],0
    start_time=time.perf_counter()
    for epoch in range(1,settings['epochs_max']+1):
        model.train()
        order=list(range(len(records)))
        random.Random(settings['seed']+epoch).shuffle(order)
        losses=[]
        for offset in range(0,len(order),16):
            indices=order[offset:offset+16]
            optimizer.zero_grad(set_to_none=True)
            for micro in range(0,len(indices),4):
                batch=[records[i] for i in indices[micro:micro+4]]
                x=tokenizer.pad([b[0] for b in batch],padding=True,return_tensors='pt').to('cuda')
                tasks=torch.tensor([b[1] for b in batch],device='cuda')
                labels=torch.tensor([b[2] for b in batch],device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):
                    outputs=model(**x)
                    loss=task_loss(outputs,tasks,labels,weights,len(indices))
                scaler.scale(loss).backward()
                losses.append(loss.detach().item())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            previous=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale()>=previous:
                scheduler.step()
                successful+=1
            if offset%1600==0:
                print(json.dumps({'stage':selection,'epoch':epoch,'rows':min(offset+16,len(order)),
                                  'total':len(order),'seconds':round(time.perf_counter()-start_time)}),flush=True)
        reports={}
        for name in (('validation',) if selection=='initial' else ('validation','safety_development')):
            values=predictions(model,tokenizer,eval_encoded[name])
            reports[name]=evaluate(evals[name],values)
            write(folder/f'epoch-{epoch}.{name}.predictions.json',[dict(r,prediction=p) for r,p in zip(evals[name],values)])
        result=reports['validation']
        rank=initial_rank(result) if selection=='initial' else safety_rank(result,reports['safety_development'])
        history.append({'epoch':epoch,'loss_micro_mean':statistics.mean(losses),'reports':reports,'rank':rank})
        write(folder/'history.json',history)
        if best is None or rank>best:
            best,stale=rank,0
            torch.save(model.state_dict(),folder/'model.pt')
            config.update(selected_epoch=epoch,selected_reports=reports)
        else:
            stale+=1
        print(json.dumps({'stage':selection,'epoch_complete':epoch,'policy_correct':result['policy_correct'],
                          'urgency_exact':result['urgency']['exact_correct'],
                          'urgent_policy_misses':len(result['urgent_policy_miss_ids']),
                          'relevance_pass_correct':result['relevance_pass_correct'],
                          'rank':rank},ensure_ascii=False),flush=True)
        if selection=='initial' and stale>=2:
            break
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda',weights_only=True))
    config.update(status='completed',epochs_completed=len(history),successful_updates=successful,
                  training_seconds=time.perf_counter()-start_time,checkpoint_sha256=digest(folder/'model.pt'))
    write(folder/'run_config.json',config)
    del optimizer,scheduler,scaler
    gc.collect()
    torch.cuda.empty_cache()
    return config


def prepare(dataset_dir=DEFAULT_DATASET):
    sources,protected,stages=load_windows_prepared(dataset_dir)
    old_initial=read(OLD_BASE/'run_config.json')
    old_safety=read(OLD_SAFETY/'run_config.json')
    _,groups,_=validation()
    evals,evaluation_hashes=load_evaluation_inputs(dataset_dir,evaluation_sets())
    protected.update(evaluation_hashes)
    paths={'safety_development':SAFETY/'development.jsonl',
           'personal100':ROOT/'filtering_training/outputs/v9_personal_use_eval_01/dataset.jsonl',
           'boundary14':ROOT/'filtering_training/outputs/v11_boundary_supplement_01/dataset.jsonl'}
    for name,path in paths.items():
        protected[str(path)]=digest(path)
    safety_manifest=read(SAFETY/'manifest.json')
    for filename,key in [('supplement.jsonl','supplement_sha256'),('development.jsonl','development_sha256')]:
        if digest(SAFETY/filename)!=safety_manifest[key]:
            raise ValueError('Frozen safety data changed')
        protected[str(SAFETY/filename)]=digest(SAFETY/filename)
    for folder,names in [(BASE,['model.safetensors','config.json','manifest.json','tokenizer.json']),
                         (OLD_BASE,['model.pt','run_config.json','tokenizer/tokenizer.json']),
                         (OLD_SAFETY,['model.pt','model-fp32.onnx','run_config.json','tokenizer/tokenizer.json'])]:
        for filename in names:
            protected[str(folder/filename)]=digest(folder/filename)
    protected[str(SAFETY/'manifest.json')]=digest(SAFETY/'manifest.json')
    protected[str(Path(__file__).resolve())]=digest(__file__)
    for task in TASKS:
        if len(sources[task])!=old_initial['train_counts'][task]:
            raise ValueError('Original task row count differs')
    replay_indices=old_safety['old_relevance_source_indices']
    expected_indices=[]
    for level in range(1,6):
        candidates=[i for i,r in enumerate(sources['relevance']) if int(r['expected']['relevance'])==level]
        expected_indices.extend(sorted(random.Random(73+level).sample(candidates,min(500,len(candidates)))))
    if replay_indices!=expected_indices:
        raise ValueError('Original relevance replay selection differs')
    supplement=load_lines(SAFETY/'supplement.jsonl')
    expected_initial=[(r,task) for task in TASKS for r in sources[task]]
    expected_safety=([(r,'urgency') for r in sources['urgency']]+
                 [(sources['relevance'][i],'relevance') for i in replay_indices]+
                 [(r,'urgency') for r in supplement]*old_safety['supplement_repeats'])
    for name, expected_rows in [('initial', expected_initial), ('safety', expected_safety)]:
        expected_values = [(r['state'], task, r['expected'][task]) for r, task in expected_rows]
        actual_values = [(r['state'], task, r['expected'][task]) for r, task in stages[name]]
        if actual_values != expected_values:
            raise ValueError(f'windows-only {name} rows changed source labels, replay indices or order')
    initial_rows, safety_rows = stages['initial'], stages['safety']
    training_bodies={body_key(r['state']) for r,_ in initial_rows+safety_rows}
    overlaps={name:[r['notification']['id'] for r in rows if body_key(r) in training_bodies] for name,rows in evals.items()}
    if any(overlaps.values()):
        raise ValueError('Training/evaluation normalized-body overlap: '+json.dumps(overlaps))
    config={'purpose':'Joint urgency5/relevance3 KLUE with windows-only relevance inputs and retained source labels',
            'created_date':created_date(),'status':'prepared','mapping':RELEVANCE_MAP,
            'input_schema':INPUT_SCHEMA,'recent_windows_policy':RECENT_WINDOWS_POLICY,
            'baseline_input_schema':LEGACY_INPUT_SCHEMA,
            'dataset_dir':str(Path(dataset_dir).resolve()),'removed_input_field':'recent_processes',
            'initial_counts':dict(Counter(task for _,task in initial_rows)),
            'safety_counts':dict(Counter(task for _,task in safety_rows)),
            'initial_relevance3_counts':dict(Counter(map_relevance(int(r['expected']['relevance'])) for r in sources['relevance'])),
            'same_source_rows_and_replay_indices':True,'new_five_level_control_trained':False,
            'initial_settings':{k:old_initial[k] for k in ('seed','epochs_max','encoder_lr','head_lr','warmup_ratio')},
            'safety_settings':{k:old_safety[k] for k in ('seed','epochs_max','encoder_lr','head_lr')},
            'eval_counts':{k:len(v) for k,v in evals.items()},'normalized_body_overlap':overlaps,
            'validation_groups':groups,'protected_hashes':protected,'final_test_used':False,
            'diagnostics_used_for_selection':False,'single_seed':True,'deployment_changed':False,
            'limitations':'historical single-seed comparison; input schema and relevance classes differ; not isolated causal evidence',
            'safety_relevance_source_indices':replay_indices}
    config['safety_settings']['warmup_ratio']=.05
    return initial_rows,safety_rows,evals,config


def export_and_verify(model,folder,tokenizer,evals,expected):
    import torch
    import numpy as np
    import onnx
    import onnxruntime as ort
    path=folder/'model-fp32.onnx'
    if path.exists():
        raise ValueError('Preserve existing export')
    model=model.cpu().eval()
    model.encoder.set_attn_implementation('eager')
    sample=tokenizer(['알림 확인','현재 작업 알림'],padding=True,return_tensors='pt',return_token_type_ids=False)
    torch.onnx.export(model,(sample['input_ids'],sample['attention_mask']),str(path),
                      input_names=['input_ids','attention_mask'],output_names=['urgency_logits','relevance_logits'],
                      dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},
                                    'urgency_logits':{0:'batch'},'relevance_logits':{0:'batch'}},
                      opset_version=17,dynamo=False,external_data=False)
    onnx.checker.check_model(str(path))
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    options.add_session_config_entry('session.inter_op.allow_spinning','0')
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    mismatches=[]
    times=[]
    cpu_predictions={}
    for name,rows in evals.items():
        values=[]
        for i,row in enumerate(rows):
            start=time.perf_counter()
            scores=[]
            for head,task in enumerate(TASKS):
                encoded=tokenizer(input_text(row,task),return_tensors='np',return_token_type_ids=False,truncation=False)
                outputs=session.run(None,{k:np.asarray(v,dtype=np.int64) for k,v in encoded.items()})
                if outputs[0].shape!=(1,5) or outputs[1].shape!=(1,3):
                    raise ValueError('Unexpected joint ONNX output shape')
                scores.append(int(outputs[head][0].argmax())+1)
            pred={'urgency_score':scores[0],'relevance_score':scores[1]}
            values.append(pred)
            times.append(time.perf_counter()-start)
            if pred!=expected[name][i]:
                mismatches.append({'dataset':name,'id':row['notification']['id'],'onnx':pred,'gpu':expected[name][i]})
        cpu_predictions[name]=values
    report={'created_date':created_date(),'input_schema':INPUT_SCHEMA,'onnx_sha256':digest(path),'onnx_bytes':path.stat().st_size,
            'cases':sum(len(v) for v in evals.values()),'score_parity':not mismatches,'mismatches':mismatches,
            'output_classes':{'urgency':5,'relevance':3},'policy':'urgency >= 4 OR relevance == 3',
            'mean_ms':statistics.mean(times)*1000,'p95_ms':float(np.percentile(times,95))*1000,
            'timing_scope':'JSON/tokenization plus two CPU calls; no warmup exclusion; excludes load/queue/UI',
            'existing_csharp_adapter_requires_output_and_policy_update':True}
    write(folder/'export.json',report)
    if mismatches:
        raise ValueError('ONNX/GPU score mismatch')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--dataset-dir',type=Path,default=DEFAULT_DATASET)
    args=parser.parse_args()
    if args.run.exists():
        raise ValueError('Preserve previous runs; choose a fresh path')
    initial,safety,evals,config=prepare(args.dataset_dir)
    print(json.dumps({k:v for k,v in config.items() if k not in ('protected_hashes','validation_groups','safety_relevance_source_indices')},ensure_ascii=False),flush=True)
    if args.dry_run:
        return
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise ValueError('CUDA required')
    torch.set_num_threads(4)
    args.run.mkdir(parents=True)
    config.update(status='running',gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                  transformers=__import__('transformers').__version__)
    write(args.run/'run_config.json',config)
    tokenizer=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
    # Verify observed input tokenization matches the deployed tokenizer.
    deployed_tokenizer=AutoTokenizer.from_pretrained(str(OLD_SAFETY/'tokenizer'),local_files_only=True)
    eval_encoded={}
    for name,rows in evals.items():
        texts=[input_text(r,task) for r in rows for task in TASKS]
        encoded,_=encode_checked(tokenizer,texts)
        old_encoded,_=encode_checked(deployed_tokenizer,texts)
        if encoded!=old_encoded:
            raise ValueError('Current and original tokenizers differ')
        eval_encoded[name]=encoded
    comparison={'existing_urgency5_relevance5':{}}
    legacy_evals=evaluation_sets(schema=LEGACY_INPUT_SCHEMA)
    baseline_values={}
    for name,rows in evals.items():
        values=baseline(legacy_evals[name],tokenizer,OLD_SAFETY)
        baseline_values[name]=values
        comparison['existing_urgency5_relevance5'][name]=evaluate(rows,values)
        write(args.run/f'baseline-{name}.predictions.json',[dict(r,prediction=p) for r,p in zip(legacy_evals[name],values)])
    if comparison['existing_urgency5_relevance5']['validation']['policy_correct']!=68:
        raise ValueError('Existing deployed baseline does not reproduce 68/74')
    write(args.run/'comparison.json',comparison)
    evaluation_folder=args.run/'evaluation-inputs'
    evaluation_folder.mkdir()
    for name, rows in evals.items():
        path=evaluation_folder/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    write(evaluation_folder/'manifest.json',{'created_date':created_date(),'input_schema':INPUT_SCHEMA,
          'purpose':'Evaluation source snapshots for the windows-only input contract; labels unchanged',
          'files':{name:{'rows':len(rows),'sha256':digest(evaluation_folder/f'{name}.jsonl')} for name,rows in evals.items()}})
    encoded_records={}
    mapped=args.run/'mapped-training'
    mapped.mkdir()
    for name,rows in [('initial',initial),('safety',safety)]:
        texts=[input_text(r['state'],task) for r,task in rows]
        encoded,lengths=encode_checked(tokenizer,texts)
        labels=[int(r['expected'][task]) if task=='urgency' else map_relevance(int(r['expected'][task])) for r,task in rows]
        encoded_records[name]=[(e,TASKS.index(task),label-1) for e,(_,task),label in zip(encoded,rows,labels)]
        with (mapped/f'{name}.jsonl').open('w',encoding='utf-8') as stream:
            for (row,task),text,label in zip(rows,texts,labels):
                stream.write(json.dumps({'state':row['state'],'task':task,'original_score':int(row['expected'][task]),
                                         'score':label,'input_text':text},ensure_ascii=False)+'\n')
        config[f'{name}_max_tokens']=max(lengths)
    write(mapped/'manifest.json',{'created_date':created_date(),'input_schema':INPUT_SCHEMA,'mapping':RELEVANCE_MAP,
                                  'files':{name:{'sha256':digest(mapped/f'{name}.jsonl'),'rows':len(rows)}
                                           for name,rows in [('initial',initial),('safety',safety)]}})
    seed=config['initial_settings']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model=build_model().cuda()
    initial_config=train_stage(args.run/'initial',model,tokenizer,encoded_records['initial'],evals,eval_encoded,
                               config['initial_settings'],'initial')
    parent_predictions=predictions(model,tokenizer,eval_encoded['validation'])
    write(args.run/'safety-parent-validation.json',evaluate(evals['validation'],parent_predictions))
    # Recreate then restore the model, as the original safety continuation did,
    # preserving the seed-73 initialization RNG consumption before dropout training.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(config['safety_settings']['seed'])
    model=build_model(base=args.run/'initial'/'encoder',pretrained=False,attention='sdpa')
    model.load_state_dict(torch.load(args.run/'initial'/'model.pt',map_location='cpu',weights_only=True))
    model=model.cuda()
    safety_config=train_stage(args.run/'safety',model,tokenizer,encoded_records['safety'],evals,eval_encoded,
                              config['safety_settings'],'safety')
    values={name:predictions(model,tokenizer,encoded) for name,encoded in eval_encoded.items()}
    comparison['new_urgency5_relevance3']={name:evaluate(evals[name],preds) for name,preds in values.items()}
    for name,preds in values.items():
        write(args.run/'safety'/f'{name}.predictions.json',[dict(r,prediction=p) for r,p in zip(evals[name],preds)])
    write(args.run/'comparison.json',comparison)
    export_report=export_and_verify(model,args.run/'safety',tokenizer,evals,values)
    for path,expected in config['protected_hashes'].items():
        if digest(Path(path))!=expected:
            raise ValueError('Protected source changed: '+path)
    config.update(status='completed',protected_sources_unchanged=True,
                  initial_selected_epoch=initial_config['selected_epoch'],safety_selected_epoch=safety_config['selected_epoch'],
                  candidate_checkpoint_sha256=safety_config['checkpoint_sha256'],onnx=export_report)
    write(args.run/'run_config.json',config)
    print(json.dumps({'completed':str(args.run),'comparison':comparison},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
