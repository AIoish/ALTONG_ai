"""용도: 사전 고정한 대비 비중 0·5·10%를 비교하고 유망 설정만 기준과 함께 3개 시드로 확인한다.
생성일: 2026-11-15
"""

import argparse
from copy import deepcopy
from pathlib import Path
import statistics

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.modeling.relevance_only import DEFAULTS, prepare, configure_mixture, rank, train
from filtering_training.preparation.migrate_relevance_policy import rescore
from filtering_training.quality.relevance_evaluation import CONTRACT, decide, metrics

DATA = ROOT/'filtering_training/outputs/v21_relevance_reaction_policy_01'
OLD = ROOT/'filtering_training/outputs/runs/2026-11-12-klue-relevance-only-03'


def study(output, *, allow_candidates=False, dry_run=False):
    output=Path(output)
    if output.exists():raise ValueError('preserve existing study; choose a new folder')
    output.mkdir(parents=True)
    plan={'purpose':'Controlled loss mixture development experiment; no independent real final test',
          'created_date':created_date(),'shares':[0.,.05,.10],'screen_seed':42,'repeat_seeds':[43,44],
          'contract':CONTRACT,'settings':DEFAULTS,'same_rows_order_schedule':True,
          'promising':'Pass both baseline-0 and model03 gates; improve exact accuracy or conflict count within the confirmed 0.5pp tolerance.',
          'tie':'Higher exact accuracy, macro F1, lower MAE; smaller supplement share on equal ranks.',
          'repeats':'Only the best promising share plus baseline-0; no promising share means no repeats.'}
    write(output/'plan.json',plan)
    rows,encoded,tokenizer,config=prepare(DATA/'current-priority',allow_candidates=allow_candidates)
    snapshots=set(config['code_hashes'])|{str(Path(__file__).resolve().relative_to(ROOT)), 'filtering_training/quality/relevance_evaluation.py'}
    for name in snapshots:
        target=output/'executed-code'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    write(output/'executed-code/manifest.json',{'purpose':'Executed training and adoption code snapshot','created_date':created_date(),
          'files':{n:digest(output/'executed-code'/n) for n in snapshots}})
    configs={}
    for share in plan['shares']:
        cfg=deepcopy(config)
        configure_mixture(rows,cfg,DATA/'baseline',share)
        configs[share]=cfg
    write(output/'mixtures.json',{str(s):c['training_mixture'] for s,c in configs.items()})
    if dry_run:return {'status':'dry_run_passed','rows':{s:len(r) for s,r in rows.items()}}
    rescore(DATA/'current-priority',OLD,output/'model03-rescore.json')
    cached={p['id']:p['predicted'] for p in load_lines(OLD/'validation.predictions.jsonl')}
    old=metrics(rows['validation'],[cached[r['id']] for r in rows['validation']])
    results=[]
    def run(share,seed):
        folder=output/f'share-{round(share*100):02d}-seed-{seed}'
        settings=dict(DEFAULTS,seed=seed)
        cfg=deepcopy(configs[share]);cfg['study_plan_sha256']=digest(output/'plan.json')
        train(folder,rows,encoded,tokenizer,cfg,settings)
        cached={p['id']:p['predicted'] for p in load_lines(folder/'validation.predictions.jsonl')}
        report=metrics(rows['validation'],[cached[r['id']] for r in rows['validation']])
        result={'share':share,'seed':seed,'run':str(folder.resolve()),'report':report,
                'checkpoint_sha256':read(folder/'run_config.json')['checkpoint_sha256']}
        results.append(result);write(output/'results.json',results)
        print({'completed':folder.name,'correct':report['exact_correct'],'misses':report['relevant_misses'],
               'false_relevant':report['false_relevant']},flush=True)
        return result
    baseline=run(0.,42)
    promising=[]
    for share in (.05,.10):
        candidate=run(share,42)
        report=candidate['report'];base=baseline['report']
        candidate['against_zero']=decide(base,report);candidate['against_model03']=decide(old,report)
        conflict=lambda r:r['by_case_kind'].get('current_history_conflict',{}).get('exact_correct',0)
        if (candidate['against_zero']['development_gate']=='pass' and candidate['against_model03']['development_gate']=='pass'
            and (report['exact_correct']>base['exact_correct'] or conflict(report)>conflict(base))):
            promising.append(candidate)
    best=max(promising,key=lambda r:rank(r['report'])) if promising else None
    write(output/'screening.json',{'baseline':baseline,'candidates':results[1:],'selected_share':best['share'] if best else None,
          'no_promising_configuration':not bool(best)})
    if best:
        for seed in (43,44):
            run(0.,seed);run(best['share'],seed)
    summary={'purpose':plan['purpose'],'created_date':created_date(),'selected_share':best['share'] if best else None,
             'model03':old,'groups':{},'paired_decisions':[], 'deployment_approved':False,'real_final_evaluation':'missing'}
    for share in sorted({r['share'] for r in results}):
        values=[r['report']['exact_accuracy'] for r in results if r['share']==share]
        summary['groups'][str(share)]={'runs':len(values),'mean_accuracy':statistics.mean(values),
            'minimum_accuracy':min(values),'maximum_accuracy':max(values),'std_accuracy':statistics.pstdev(values)}
    if best:
        for seed in (42,43,44):
            base=next(r for r in results if r['share']==0 and r['seed']==seed)
            candidate=next(r for r in results if r['share']==best['share'] and r['seed']==seed)
            summary['paired_decisions'].append({'seed':seed,'against_zero':decide(base['report'],candidate['report']),
                                               'against_model03':decide(old,candidate['report'])})
    write(output/'results.json',results);write(output/'summary.json',summary)
    write(output/'status.json',{'status':'completed','runs':len(results),'training_ready_promoted':False})
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-candidate-labels',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    print(study(args.output,allow_candidates=args.allow_candidate_labels,dry_run=args.dry_run),flush=True)
