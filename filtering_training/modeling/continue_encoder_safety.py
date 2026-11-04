"""용도: 보존된 base 체크포인트에 별도 안전 보강과 기존 자료 재학습을 적용한다.
생성일: 2026-10-10
"""
import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from functools import partial
from filtering_training.modeling.encoder_score_experiment import build_model, score
from filtering_training.common.encoder_input import LEGACY_INPUT_SCHEMA, legacy_input_text as input_text, encode_checked
from filtering_training.common.validation_data import validation as current_validation

# Historical 5/5 continuation retains its input contract; new training uses joint_urgency5_relevance3.
validation = partial(current_validation, schema=LEGACY_INPUT_SCHEMA)
from src.filtering.policy import should_pass

ROOT = Path(__file__).resolve().parents[2]
TASKS=('urgency','relevance')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--supplement',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.run.exists():
        raise ValueError('Use a new run directory')
    if args.epochs not in (1,2,3):
        raise ValueError('Epochs must be 1..3')
    random.seed(73); np.random.seed(73); torch.manual_seed(73); torch.set_num_threads(4)
    cfg=read(args.parent/'run_config.json')
    if digest(args.parent/'model.pt')!=cfg['checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    prepared=ROOT/'filtering_training/outputs/v7_relevance_ladder_01/prepared_laya_v2'
    manifest=read(args.supplement/'manifest.json')
    supplement=lines(args.supplement/'supplement.jsonl')
    dev=lines(args.supplement/'development.jsonl')
    if digest(args.supplement/'supplement.jsonl')!=manifest['supplement_sha256'] or digest(args.supplement/'development.jsonl')!=manifest['development_sha256']:
        raise ValueError('Frozen supplement/development changed')
    originals={task:lines(prepared/task/'train.jsonl') for task in TASKS}
    # All previous urgency cases retained; balance existing relevance replay across five scores.
    replay_relevance=[]
    replay_indices=[]
    for level in range(1,6):
        candidates=[i for i,r in enumerate(originals['relevance']) if int(r['expected']['relevance'])==level]
        selected=sorted(random.Random(73+level).sample(candidates,min(500,len(candidates))))
        replay_indices.extend(selected)
        replay_relevance.extend(originals['relevance'][i] for i in selected)
    rows=[(r,0,False) for r in originals['urgency']]+[(r,1,False) for r in replay_relevance]+[(r,0,True) for r in supplement]*4
    tok=AutoTokenizer.from_pretrained(args.parent/'tokenizer',local_files_only=True)
    tokenized,lengths=encode_checked(tok,[input_text(r['state'],TASKS[task]) for r,task,_ in rows])
    records=[(e,task,int(r['expected'][TASKS[task]])-1,new) for e,(r,task,new) in zip(tokenized,rows)]
    old_samples,_,old_texts=validation()
    old_encoded,_=encode_checked(tok,[text for pair in old_texts for text in pair])
    dev_encoded,_=encode_checked(tok,[input_text(r,task) for r in dev for task in TASKS])
    sources=[prepared/task/'train.jsonl' for task in TASKS]
    sources += [args.parent/'model.pt',args.parent/'model-fp32.onnx',args.parent/'tokenizer/tokenizer.json',
                args.parent/'run_config.json',args.supplement/'supplement.jsonl',args.supplement/'development.jsonl',
                ROOT/'filtering_training/outputs/v9_personal_use_eval_01/dataset.jsonl',
                ROOT/'filtering_training/outputs/runs/2026-10-10-base-personal-eval-01/report.json']
    protected={str(p):digest(p) for p in sources}
    config={'purpose':'Continue existing base with separate safety supplement and previous-data replay',
            'created_date':'2026-10-10','status':'prepared','parent_run':str(args.parent),
            'parent_checkpoint_sha256':cfg['checkpoint_sha256'],'seed':73,'epochs_max':args.epochs,
            'encoder_lr':5e-6,'head_lr':2e-5,'effective_batch':16,'micro_batch':4,
            'loss':'five-class CE, equal nominal total task contributions; no threshold objective',
            'supplement_unique_rows':len(supplement),'supplement_repeats':4,
            'old_urgency_rows':len(originals['urgency']),'old_relevance_replay_rows':len(replay_relevance),
            'old_relevance_source_indices':replay_indices,'train_counts':dict(Counter(TASKS[t] for _,t,_,_ in records)),
            'train_token_max':max(lengths),'truncated_rows':0,'development_rows':len(dev),'original_validation_rows':len(old_samples),
            'prior_100_eval_used_for_training':False,'prior_100_eval_used_for_checkpoint_selection':False,
            'checkpoint_selection':'Prefer eligible checkpoints (original policy >=65/74 and urgent misses <=2); then development urgent misses, original urgent misses, development false passes, original policy correct, development policy correct',
            'final_test_used':False,'label_status':'AI proposed, user authorized supplement; no individual human approval asserted',
            'protected_hashes':protected,'optimizer_resumed':False,'executed_code_sha256':digest(Path(__file__))}
    if args.dry_run:
        print(json.dumps({k:v for k,v in config.items() if k not in ('protected_hashes','old_relevance_source_indices')},ensure_ascii=False),flush=True)
        return
    if not torch.cuda.is_available():
        raise ValueError('GPU required for this continuation')
    args.run.mkdir(parents=True)
    config['status']='training'
    write(args.run/'run_config.json',config)
    model=build_model(AutoConfig.from_pretrained(args.parent/'encoder',local_files_only=True),attention='sdpa')
    model.load_state_dict(torch.load(args.parent/'model.pt',map_location='cpu',weights_only=True))
    model.cuda(); model.encoder.gradient_checkpointing_enable()
    tok.save_pretrained(args.run/'tokenizer'); model.encoder.config.save_pretrained(args.run/'encoder')

    def predictions(encoded):
        model.eval(); logits=[]
        with torch.inference_mode():
            for offset in range(0,len(encoded),8):
                x=tok.pad(encoded[offset:offset+8],padding=True,return_tensors='pt').to('cuda')
                logits.extend(model(**x).float().cpu().numpy())
        return [{task+'_score':int(logits[2*i+j][j].argmax())+1 for j,task in enumerate(TASKS)} for i in range(len(encoded)//2)]

    def dev_metrics(preds):
        rows=[{'gold':r['label'],'prediction':p} for r,p in zip(dev,preds)]
        return {'count':len(rows),'urgent_count':sum(r['gold']['urgency_score']>=4 for r in rows),
                'urgent_misses':sum(r['gold']['urgency_score']>=4 and p['urgency_score']<4 and p['relevance_score']<4 for r,p in zip(rows,preds)),
                'false_passes':sum(not should_pass(r['gold']['urgency_score'],r['gold']['relevance_score']) and should_pass(p['urgency_score'],p['relevance_score']) for r,p in zip(rows,preds)),
                'policy_correct':sum(should_pass(r['gold']['urgency_score'],r['gold']['relevance_score'])==should_pass(p['urgency_score'],p['relevance_score']) for r,p in zip(rows,preds))}

    original_baseline=predictions(old_encoded)
    old_baseline_metrics=score(old_samples,original_baseline)
    if old_baseline_metrics['policy_correct']!=67 or old_baseline_metrics['urgent_false_blocks']!=2:
        raise ValueError('Parent checkpoint baseline does not reproduce expected original validation')
    development_baseline=predictions(dev_encoded)
    write(args.run/'baseline.json',{'original':old_baseline_metrics,'development':dev_metrics(development_baseline)})
    print(json.dumps({'baseline_original':old_baseline_metrics['policy_correct'],'baseline_dev':dev_metrics(development_baseline)},ensure_ascii=False),flush=True)
    counts=[sum(task==i for _,task,_,_ in records) for i in range(2)]
    weights=torch.tensor([len(records)/(2*n) for n in counts],device='cuda')
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':5e-6},{'params':model.heads.parameters(),'lr':2e-5}],weight_decay=.01,foreach=False)
    updates=math.ceil(len(records)/16)*args.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,max(1,int(updates*.05)),updates)
    scaler=torch.amp.GradScaler('cuda')
    history=[]; best=None; total_updates=0; start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); order=list(range(len(records))); random.Random(73+epoch).shuffle(order); losses=[]
        for offset in range(0,len(order),16):
            ids=order[offset:offset+16]; optimizer.zero_grad(set_to_none=True)
            for micro in range(0,len(ids),4):
                batch=[records[i] for i in ids[micro:micro+4]]
                x=tok.pad([b[0] for b in batch],padding=True,return_tensors='pt').to('cuda')
                tasks=torch.tensor([b[1] for b in batch],device='cuda'); labels=torch.tensor([b[2] for b in batch],device='cuda')
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(**x)[torch.arange(len(batch),device='cuda'),tasks]
                    ce=torch.nn.functional.cross_entropy(logits.float(),labels,reduction='none')
                    loss=(ce*weights[tasks]).sum()/len(ids)
                scaler.scale(loss).backward(); losses.extend(ce.detach().cpu().tolist())
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            previous=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale()>=previous:
                scheduler.step();total_updates+=1
            if offset%800==0:
                print(json.dumps({'epoch':epoch,'rows':min(offset+16,len(records)),'total':len(records),'recent_ce':sum(losses[-800:])/len(losses[-800:]),'elapsed_seconds':round(time.perf_counter()-start)}),flush=True)
        old_preds=predictions(old_encoded); dm_preds=predictions(dev_encoded)
        om=score(old_samples,old_preds); dm=dev_metrics(dm_preds)
        eligible=om['policy_correct']>=65 and om['urgent_false_blocks']<=2
        rank=(int(eligible),-dm['urgent_misses'],-om['urgent_false_blocks'],-dm['false_passes'],om['policy_correct'],dm['policy_correct'])
        result={'epoch':epoch,'train_ce':sum(losses)/len(losses),'original':om,'development':dm,'eligible':eligible,'rank':rank,'elapsed_seconds':time.perf_counter()-start}
        history.append(result);write(args.run/'history.json',history);write(args.run/f'epoch-{epoch}.json',result)
        write(args.run/f'epoch-{epoch}.predictions.json',[{'validation_index':i,'notification_id':s.notification.id,
                                                        'notification':s.notification.model_dump(mode='json'),
                                                        'context':s.context.model_dump(mode='json'),'gold':s.label.model_dump(mode='json'),'prediction':p}
                                                       for i,(s,p) in enumerate(zip(old_samples,old_preds))])
        write(args.run/f'epoch-{epoch}.development.predictions.json',[dict(r,prediction=p) for r,p in zip(dev,dm_preds)])
        if best is None or rank>best:
            best=rank
            torch.save(model.state_dict(),args.run/'model.pt')
            config.update(selected_epoch=epoch,selected_metrics=om,selected_development=dm,selected_eligible=eligible)
        print(json.dumps({'epoch_complete':epoch,'original_policy':om['policy_correct'],'original_urgent_misses':om['urgent_false_blocks'],'development':dm,'eligible':eligible},ensure_ascii=False),flush=True)
    for path,expected in protected.items():
        if digest(Path(path))!=expected:
            raise ValueError('Protected artifact changed: '+path)
    config.update(status='completed',training_seconds=time.perf_counter()-start,epochs_completed=len(history),
                  successful_updates=total_updates,checkpoint_sha256=digest(args.run/'model.pt'),protected_sources_unchanged=True)
    write(args.run/'run_config.json',config)
    print(json.dumps({'complete':str(args.run),'selected_epoch':config['selected_epoch'],'selected_eligible':config['selected_eligible'],'seconds':config['training_seconds']}),flush=True)

if __name__=='__main__':
    main()
