"""용도: 안전 보강 base에 경계·관련도 보강을 적용하고 부모 모델보다 나쁜 후보를 제외한다.
생성일: 2026-10-10
"""
import argparse
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
from filtering_training.common.artifact_io import ROOT, read, write, digest, load_lines as lines
from filtering_training.common.encoder_training import predict_logits, train_epoch
from filtering_training.modeling.encoder_score_experiment import build_model, score
from filtering_training.common.encoder_input import LEGACY_INPUT_SCHEMA, legacy_input_text as input_text, encode_checked
from filtering_training.common.validation_data import validation as current_validation

validation = partial(current_validation, schema=LEGACY_INPUT_SCHEMA)
from src.filtering.policy import should_pass

TASKS=('urgency','relevance')


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
    rows=[(r,0,False) for r in originals['urgency']]+[(r,1,False) for r in replay_relevance]+[(r,TASKS.index(task),True) for r in supplement for task in r['expected']]*4
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
            'checkpoint_selection':'Parent is epoch 0 candidate; require original policy and urgency threshold misses no worse than parent; development policy misses/false passes/urgency threshold misses no worse; then rank original urgent misses and development errors',
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
        logits = predict_logits(model, tok, encoded)
        return [{task+'_score':int(logits[j][2*i+j].argmax())+1 for j,task in enumerate(TASKS)} for i in range(len(encoded)//2)]

    def dev_metrics(preds):
        rows=[{'gold':r['label'],'prediction':p} for r,p in zip(dev,preds)]
        return {'count':len(rows),'urgent_count':sum(r['gold']['urgency_score']>=4 for r in rows),
                'urgent_threshold_misses':sum(r['gold']['urgency_score']>=4 and p['urgency_score']<4 for r,p in zip(rows,preds)),
                'urgent_misses':sum(r['gold']['urgency_score']>=4 and p['urgency_score']<4 and p['relevance_score']<4 for r,p in zip(rows,preds)),
                'false_passes':sum(not should_pass(r['gold']['urgency_score'],r['gold']['relevance_score']) and should_pass(p['urgency_score'],p['relevance_score']) for r,p in zip(rows,preds)),
                'policy_correct':sum(should_pass(r['gold']['urgency_score'],r['gold']['relevance_score'])==should_pass(p['urgency_score'],p['relevance_score']) for r,p in zip(rows,preds))}

    original_baseline=predictions(old_encoded)
    old_baseline_metrics=score(old_samples,original_baseline)
    if old_baseline_metrics['policy_correct']!=cfg['selected_metrics']['policy_correct'] or old_baseline_metrics['urgent_false_blocks']!=cfg['selected_metrics']['urgent_false_blocks']:
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
    history=[]; total_updates=0; start=time.perf_counter()
    baseline_dm=dev_metrics(development_baseline)
    def ranking(om,dm):
        return (-om['urgent_false_blocks'],-dm['urgent_misses'],-dm['false_passes'],om['policy_correct'],dm['policy_correct'])
    best=ranking(old_baseline_metrics,baseline_dm)
    torch.save(model.state_dict(),args.run/'model.pt')
    config.update(selected_epoch=0,selected_metrics=old_baseline_metrics,selected_development=baseline_dm,selected_eligible=True)
    write(args.run/'epoch-0.predictions.json',[{'validation_index':i,'notification_id':v.notification.id,'notification':v.notification.model_dump(mode='json'),'context':v.context.model_dump(mode='json'),'gold':v.label.model_dump(mode='json'),'prediction':q} for i,(v,q) in enumerate(zip(old_samples,original_baseline))])
    write(args.run/'epoch-0.development.predictions.json',[dict(r,prediction=q) for r,q in zip(dev,development_baseline)])
    for epoch in range(1,args.epochs+1):
        epoch_result = train_epoch(
            model, tok, records, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            weights=weights, seed=73 + epoch,
            progress=lambda count: print(json.dumps({'epoch': epoch, 'rows': count,
                'total': len(records), 'elapsed_seconds': round(time.perf_counter()-start)}), flush=True))
        total_updates += epoch_result['successful_updates']
        old_preds=predictions(old_encoded); dm_preds=predictions(dev_encoded)
        om=score(old_samples,old_preds); dm=dev_metrics(dm_preds)
        eligible=(om['policy_correct']>=old_baseline_metrics['policy_correct'] and om['urgent_false_blocks']<=old_baseline_metrics['urgent_false_blocks'] and dm['urgent_misses']<=baseline_dm['urgent_misses'] and dm['false_passes']<=baseline_dm['false_passes'] and dm['urgent_threshold_misses']<=baseline_dm['urgent_threshold_misses'] and dm['policy_correct']>=baseline_dm['policy_correct'])
        rank=ranking(om,dm)
        result={'epoch':epoch,'loss_micro_mean':epoch_result['loss_micro_mean'],'original':om,'development':dm,'eligible':eligible,'rank':rank,'elapsed_seconds':time.perf_counter()-start}
        history.append(result);write(args.run/'history.json',history);write(args.run/f'epoch-{epoch}.json',result)
        write(args.run/f'epoch-{epoch}.predictions.json',[{'validation_index':i,'notification_id':s.notification.id,
                                                        'notification':s.notification.model_dump(mode='json'),
                                                        'context':s.context.model_dump(mode='json'),'gold':s.label.model_dump(mode='json'),'prediction':p}
                                                       for i,(s,p) in enumerate(zip(old_samples,old_preds))])
        write(args.run/f'epoch-{epoch}.development.predictions.json',[dict(r,prediction=p) for r,p in zip(dev,dm_preds)])
        if eligible and rank>best:
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
