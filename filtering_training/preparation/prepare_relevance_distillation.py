"""용도: 실제 수집의 생성/개발 분할 동결, 교사 초안 검수·확장, 누출을 막는 학생 후보 조립을 제공한다.
생성일: 2026-11-15
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from filtering_training.common.artifact_io import created_date, digest, load_lines, read, write
from filtering_training.common.encoder_input import INPUT_SCHEMA, encode_checked, input_text
from filtering_training.common.relevance_policy import POLICY, apply_policy, targetless
from filtering_training.preparation.curate_relevance import key, write_lines
from filtering_training.preparation.prepare_relevance_experiment import audit_rows, assign_splits, group_rows, input_key, verify_splits


def freeze(source, output, *, seed=42):
    source, output = Path(source).resolve(), Path(output)
    if output.exists():
        raise ValueError('preserve existing split before teacher generation')
    protected = {str(source): digest(source)}
    originals = load_lines(source)
    held = [r for r in originals if r['score'] is None]
    rows = apply_policy([r for r in originals if r['score'] is not None])
    for row in rows:
        row['experiment'] = {'origin':'real_seed', 'train_only':False,
            'label_source':'user' if row['review'].get('human_reviewed') else 'assistant',
            'group_hints':['capture:'+row['group_id']]}
    audit = audit_rows(rows)
    if audit['exact_input_label_conflicts'] or audit['content_window_score_warnings']:
        raise ValueError('source label conflict requires review')
    groups, near = group_rows(rows, reaction_predicate=targetless)
    split = assign_splits(rows, groups, seed=seed)
    verification = verify_splits(split, rows, groups, near, reaction_predicate=targetless)
    output.mkdir(parents=True)
    write_lines(output/'generation-seeds.jsonl',[r for r in split if r['split']=='train'])
    write_lines(output/'development-heldout.jsonl',[r for r in split if r['split']=='validation'])
    write_lines(output/'previously-held.jsonl',held)
    write_lines(output/'near-links.jsonl',near)
    write(output/'verification.json',verification)
    if digest(source)!=protected[str(source)]:raise ValueError('source changed')
    write(output/'manifest.json',{'purpose':'Freeze existing capture before teacher generation; not unseen final evaluation',
        'created_date':created_date(),'policy':POLICY,'seed':seed,'training_ready':False,
        'teacher':'Codex assistant, human-reviewed policy; generated labels remain proposals',
        'protected_sources':protected,'counts':{s:dict(Counter(r['score'] for r in split if r['split']==s)) for s in ('train','validation')},
        'groups':{s:len({r['experiment_group'] for r in split if r['split']==s}) for s in ('train','validation')},
        'held_rows':len(held),'files':{p.name:digest(p) for p in output.iterdir() if p.is_file()},
        'limitations':['Same capture was previously inspected and used in historical models.',
                       'Labels mix human answers, policy application and assistant proposals.',
                       'Session grouping is provisional; lexical checks do not prove semantic independence.']})


def verify_frozen(folder):
    return verified_artifact(folder)


def verified_artifact(folder):
    folder=Path(folder).resolve();manifest=read(folder/'manifest.json')
    for name,expected in manifest['files'].items():
        path=(folder/name).resolve()
        expected=expected['sha256'] if isinstance(expected,dict) else expected
        if not path.is_relative_to(folder) or digest(path)!=expected:
            raise ValueError('artifact file changed: '+name)
    for path,expected in manifest.get('protected_sources',{}).items():
        if digest(path)!=expected:raise ValueError('protected source changed')
    return manifest


def merge_teacher(source, addition, output):
    source,addition,output=map(Path,(source,addition,output))
    if output.exists():raise ValueError('preserve previous teacher batch')
    protected={}
    for folder in (source,addition):
        verified_artifact(folder)
        for name in ('manifest.json','candidates.jsonl'):
            protected[str((folder/name).resolve())]=digest(folder/name)
    rows=load_lines(source/'candidates.jsonl')+load_lines(addition/'candidates.jsonl')
    audit=audit_rows(rows)
    if audit['exact_input_label_conflicts'] or audit['content_window_score_warnings'] or audit['exact_duplicate_excess_rows']:
        raise ValueError('teacher expansion duplicates or conflicts')
    if any(r['split']!='train' or r['score'] is None for r in rows):
        raise ValueError('only scored training candidates can be merged')
    output.mkdir(parents=True);write_lines(output/'candidates.jsonl',rows)
    write(output/'manifest.json',{'purpose':'Active teacher expansion; original reviewed rows preserved, not an additional copy',
        'created_date':created_date(),'training_ready':False,'rows':len(rows),
        'user_confirmed_rows':sum(r.get('review',{}).get('human_reviewed') is True for r in rows),
        'score_counts':dict(Counter(r['score'] for r in rows)),'protected_sources':protected,
        'files':{'candidates.jsonl':digest(output/'candidates.jsonl')}})
    return rows


def student_materials(reference, frozen, teacher, real_review, decisions=None):
    """Rebuild the candidate deterministically; validation never supplies teacher seeds."""
    reference,frozen,teacher,real_review=map(Path,(reference,frozen,teacher,real_review))
    verify_frozen(frozen)
    for folder in (teacher,real_review):verified_artifact(folder)
    seeds=load_lines(frozen/'generation-seeds.jsonl')
    original_held=load_lines(frozen/'development-heldout.jsonl')
    evaluation=load_lines(real_review/'candidates.jsonl')
    held_by_id={r['id']:r for r in original_held}
    if len(evaluation)!=len(held_by_id) or {r['id'] for r in evaluation}!=set(held_by_id):
        raise ValueError('real review must cover exactly the frozen development rows')
    for row in evaluation:
        review=row.get('review',{})
        if (row['state']!=held_by_id[row['id']]['state'] or review.get('human_reviewed') is not True or
                review.get('reviewer')!='user' or review.get('input_sha256')!=input_key(row) or
                review.get('proposed_score')!=row['score'] or row['score'] not in (1,2,3)):
            raise ValueError('real evaluation input or human attribution changed')
    originals=[r for split in ('train','validation') for r in load_lines(reference/f'relevance/{split}.jsonl')]
    old_real={r['id']:r for r in originals if r['experiment']['origin']=='real'}
    expected_real={r['id']:r for r in seeds+original_held}
    if set(old_real)!=set(expected_real) or any(old_real[k]['state']!=v['state'] for k,v in expected_real.items()):
        raise ValueError('reference real capture does not match the frozen partition')
    sources=deepcopy([r for r in originals if r['id'] not in old_real])
    for row in sources:
        # Preserve all earlier non-real lineage, even if a bridge later removes training rows.
        row['experiment']['group_hints'].append('prior-group:'+row['experiment_group'])
    for old in seeds+evaluation:
        row=deepcopy(old);is_seed=row['id'] not in held_by_id
        group=(row if is_seed else held_by_id[row['id']])['experiment_group']
        row['split']='train' if is_seed else 'validation'
        row['experiment']={'origin':'real' if is_seed else 'real_development',
            'train_only':is_seed,'label_source':'user' if row.get('review',{}).get('human_reviewed') else 'assistant',
            'group_hints':['capture:'+row['group_id'],'teacher-seed:'+group]}
        sources.append(row)
    seed_by_id={r['id']:r for r in seeds}
    for original in load_lines(teacher/'candidates.jsonl'):
        row=deepcopy(original)
        if (not row.get('seed_ids') or not set(row['seed_ids'])<=seed_by_id.keys() or
                row['split']!='train' or row['teacher'].get('human_reviewed') is not False):
            raise ValueError('teacher seed or attribution invalid')
        row['experiment']['group_hints']=['teacher-seed:'+seed_by_id[k]['experiment_group'] for k in row['seed_ids']]
        row['experiment']['train_only']=True
        row['experiment']['loss_origin']='pilot'
        sources.append(row)
    policy_changes=[];by_id={r['id']:r for r in sources}
    if len(by_id)!=len(sources):raise ValueError('duplicate student source IDs')
    selected=load_lines(decisions) if decisions is not None else []
    if len({d['id'] for d in selected})!=len(selected):raise ValueError('duplicate policy decision')
    for decision in selected:
        row=by_id.get(decision['id'])
        if row is None or row['split']!='train':raise ValueError('policy decisions apply only to training rows')
        if row.get('review',{}).get('human_reviewed') is True:raise ValueError('assistant cannot replace human gold')
        if (decision.get('reviewer')!='assistant' or decision.get('human_reviewed') is not False or
                not decision.get('reason') or decision.get('input_sha256')!=input_key(row) or
                type(decision.get('proposed_score')) is not int or decision['proposed_score'] not in (1,2,3)):
            raise ValueError('explicit valid policy decision required')
        if targetless(row['state']['notification']['body']) and decision['proposed_score']!=1:
            raise ValueError('targetless policy conflict')
        policy_changes.append(dict(id=row['id'],previous_score=row['score'],score=decision['proposed_score'],input_sha256=input_key(row)))
        row['review']=dict(deepcopy(decision),previous_review=deepcopy(row.get('review',{})))
        row['score']=decision['proposed_score'];row['review_status']='assistant_policy_aligned_candidate'
        row['experiment']['label_source']='assistant'
    # Group before discarding anything, so indirect lineage/near-body bridges are removed too.
    groups,near=group_rows(sources,reaction_predicate=targetless)
    validation_groups={groups[r['id']] for r in sources if r['split']=='validation'}
    excluded=[];kept=[]
    for row in sources:
        if row['split']=='train' and groups[row['id']] in validation_groups:
            if row['experiment']['origin']=='teacher_distilled':raise ValueError('teacher overlaps development evaluation')
            excluded.append({'id':row['id'],'reason':'validation_group_input_body_or_near_overlap','input_sha256':input_key(row)})
            continue
        row['experiment_group']=groups[row['id']];row['training_eligible']=False;kept.append(row)
    # Real development rows previously in training are moved, never retained under the old split.
    excluded.extend({'id':r['id'],'reason':'real_development_moved_out_of_training','input_sha256':input_key(r)} for r in evaluation)
    ids={r['id'] for r in kept};near=[e for e in near if e['left'] in ids and e['right'] in ids]
    audit=audit_rows(kept)
    if audit['exact_input_label_conflicts'] or audit['content_window_score_warnings'] or audit['exact_duplicate_excess_rows']:
        raise ValueError('student candidate input duplicates or label conflicts')
    verification=verify_splits(kept,kept,{r['id']:r['experiment_group'] for r in kept},near,reaction_predicate=targetless)
    return kept,excluded,near,policy_changes,verification


def prepare_student(reference, frozen, teacher, real_review, output, decisions=None):
    from filtering_training.quality.verify_relevance_experiment import verify
    reference,frozen,teacher,real_review,output=map(Path,(reference,frozen,teacher,real_review,output))
    if output.exists():raise ValueError('preserve existing student candidate')
    verify(reference,write_report=False)
    rows,excluded,near,changes,verification=student_materials(reference,frozen,teacher,real_review,decisions)
    paths={'reference':str(reference.resolve()),'frozen':str(frozen.resolve()),
           'teacher':str(teacher.resolve()),'real_review':str(real_review.resolve()),
           'decisions':str(Path(decisions).resolve()) if decisions is not None else None}
    protected={}
    for folder in (reference,frozen,teacher,real_review):
        manifest=read(folder/'manifest.json');protected[str((folder/'manifest.json').resolve())]=digest(folder/'manifest.json')
        for name in manifest['files']:protected[str((folder/name).resolve())]=digest(folder/name)
    if decisions is not None:protected[str(Path(decisions).resolve())]=digest(decisions)
    output.mkdir(parents=True);(output/'relevance').mkdir();(output/'review').mkdir()
    for split in ('train','validation'):write_lines(output/f'relevance/{split}.jsonl',[r for r in rows if r['split']==split])
    for name,values in [('excluded',excluded),('near-duplicate-links',near),('policy-changes',changes)]:write_lines(output/f'review/{name}.jsonl',values)
    write(output/'review/verification.json',verification)
    write(output/'manifest.json',{'purpose':'Real-seed distillation student candidate; development selection only, fresh KLUE base required',
        'created_date':created_date(),'input_schema':INPUT_SCHEMA,'training_ready':False,'label_policy':POLICY,
        'distillation_preparation':paths,'protected_sources':protected,'fresh_base_required':True,
        'splits':{s:{'rows':sum(r['split']==s for r in rows),'origins':dict(Counter(r['experiment']['origin'] for r in rows if r['split']==s))} for s in ('train','validation')},
        'limitations':['Real development capture was previously trained and inspected; not an independent final test.',
                       'New name-only policy and teacher labels are development choices; AI-reviewed rows are not human gold.',
                       'Real capture partition is provisional and cannot prove session independence.'],
        'files':{p.relative_to(output).as_posix():{'sha256':digest(p)} for p in output.rglob('*') if p.is_file()}})
    return verify(output,write_report=False)


def import_teacher(folder, candidates_path, output):
    from transformers import AutoTokenizer
    folder,candidates_path,output=map(Path,(folder,candidates_path,output))
    if output.exists():raise ValueError('preserve prior teacher batch')
    verify_frozen(folder)
    seeds={r['id']:r for r in load_lines(folder/'generation-seeds.jsonl')}
    held=load_lines(folder/'development-heldout.jsonl')+load_lines(folder/'previously-held.jsonl')
    rows=load_lines(candidates_path)
    if not rows:raise ValueError('empty teacher batch')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate candidate IDs')
    for row in rows:
        teacher=row.get('teacher',{})
        if teacher.get('reviewer')!='assistant' or teacher.get('human_reviewed') is not False or not teacher.get('reason'):
            raise ValueError('teacher proposals must retain assistant attribution and rationale')
        if not row.get('seed_ids') or not set(row['seed_ids'])<=seeds.keys():
            raise ValueError('teacher used an unknown or heldout seed')
        if row['review_status']!='teacher_proposed_unreviewed' or row['split']!='train':
            raise ValueError('teacher rows are unapproved training candidates only')
        if targetless(row['state']['notification']['body']) and row['score']!=1:
            raise ValueError('targetless policy conflict')
        row['experiment']={'origin':'teacher_distilled','train_only':True,'label_source':'assistant',
            'group_hints':['teacher-seed:'+seeds[s]['experiment_group'] for s in row['seed_ids']]}
    audit=audit_rows(rows)
    if audit['exact_input_label_conflicts'] or audit['content_window_score_warnings'] or audit['exact_duplicate_excess_rows']:
        raise ValueError('teacher input duplicates or score conflicts')
    previous=[]
    for old in held:
        r=deepcopy(old);r['id']='held:'+r['id'];r['experiment']={'group_hints':['held:'+str(r.get('experiment_group',r['group_id']))]}
        previous.append(r)
    if len({r['id'] for r in previous+rows})!=len(previous+rows):raise ValueError('ID collision')
    groups,near=group_rows(previous+rows,reaction_predicate=targetless)
    held_groups={groups[r['id']] for r in previous}
    if any(groups[r['id']] in held_groups for r in rows):raise ValueError('teacher candidate overlaps heldout input/body/near group')
    for row in rows:row['experiment_group']=groups[row['id']]
    tokenizer_path=Path(__file__).resolve().parents[1]/'outputs/models/klue-roberta-base'
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True)
    _,lengths=encode_checked(tokenizer,[input_text(r['state'],'relevance') for r in rows])
    output.mkdir(parents=True)
    write_lines(output/'candidates.jsonl',rows)
    write(output/'manifest.json',{'purpose':'Real-seed teacher proposals, not human gold or approved training',
        'created_date':created_date(),'policy':POLICY,'teacher':'Codex assistant','training_ready':False,
        'frozen_manifest_sha256':digest(folder/'manifest.json'),'input_candidates_sha256':digest(candidates_path),
        'rows':len(rows),'score_counts':dict(Counter(r['score'] for r in rows)),
        'case_counts':dict(Counter(r['case_kind'] for r in rows)),'maximum_tokens':max(lengths),
        'files':{'candidates.jsonl':digest(output/'candidates.jsonl')},'heldout_lexical_overlap':False,
        'limitations':['Teacher already saw historical capture context; this is not a blind final benchmark.',
                       'Synthetic analogues are not real captures; do not infer calibrated confidence from rationale.']})
    return rows


def prepare_review(source, output, *, first=None):
    from filtering_training.review.prepare_real_review_packets import build_packet, write_packet
    source,output=Path(source),Path(output)
    verified_artifact(source)
    rows=load_lines(source/'candidates.jsonl')
    adapters=[]
    for i,r in enumerate(rows,1):
        row=deepcopy(r);row['source_number']=str(i)
        row['review']={'input_sha256':input_key(row)}
        row['provenance']={'received_kst':(datetime.fromisoformat(row['state']['notification']['timestamp'].replace('Z','+00:00'))+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'),
                           'quality_flags':['synthetic_teacher_proposal_not_real_capture']}
        adapters.append(row)
    # Representative relation types first, then the full batch. IDs remain tied to exact inputs.
    first=first if first is not None else list(range(1,min(len(rows),7)+1))
    if not first or len(first)>26 or len(set(first))!=len(first) or any(not 1<=i<=len(rows) for i in first):
        raise ValueError('valid unique representative card positions required')
    rest=[i for i in range(1,len(rows)+1) if i not in first]
    cases=[]
    for i,number in enumerate(first+rest):
        row=rows[number-1]
        cases.append({'case_id':chr(65+i) if i<len(first) else f'R{number:03d}',
            'round':1 if i<len(first) else 2+(i-len(first))//16,'source_numbers':[str(number)],
            'title':row['scenario']+' / '+row['case_kind'],
            'question':'이 입력에서 교사 제안 점수가 맞는지 확인해 줘.', 'ambiguity':row['teacher']['reason']})
    packet=build_packet(adapters,{'cases':cases,'quality_labels':{'synthetic_teacher_proposal_not_real_capture':'실제 CSV의 패턴에서 만든 합성 교사 초안'},'previously_held_reason':'__none__'})
    packet.pop('packet_id')
    packet.update(instructions='먼저 1차 A~G의 방향을 확인해 줘. 나머지는 2차 이후에 있어. 미응답은 확정되지 않아.',
        overview='실제 CSV의 생성용 사례를 바탕으로 만든 교사 초안이다. 실제 알림 원문·사용자 확정 정답이 아니며 아직 학습하지 않았다.',
        source_candidates_sha256=digest(source/'candidates.jsonl'))
    packet['packet_id']=key(packet)
    return write_packet(output,packet,source_candidates_sha256=digest(source/'candidates.jsonl'))


def apply_teacher_answers(source, review_folder, answers_path, output):
    from filtering_training.review.prepare_real_review_packets import answer_decisions
    source,review_folder,answers_path,output=map(Path,(source,review_folder,answers_path,output))
    if output.exists():raise ValueError('preserve previous teacher review')
    for folder in (source,review_folder):verified_artifact(folder)
    packet=read(review_folder/'packet.json')
    if packet['source_candidates_sha256']!=digest(source/'candidates.jsonl'):
        raise ValueError('review belongs to another teacher batch')
    rows=load_lines(source/'candidates.jsonl')
    decisions=[deepcopy(r.get('review',{'id':r['id'],'input_sha256':input_key(r),'proposed_score':r['score'],
                    'reviewer':'assistant','human_reviewed':False})) for r in rows]
    updated=answer_decisions(packet,read(answers_path),decisions)
    for row,old,decision in zip(rows,decisions,updated):
        if decision==old:continue
        score=decision['proposed_score']
        if targetless(row['state']['notification']['body']) and score not in (None,1):
            raise ValueError('answer conflicts with confirmed reaction policy')
        row['score']=score;row['review']=decision
        row['review_status']='user_confirmed_teacher_example' if score is not None else 'user_held_teacher_example'
        row['experiment']['label_source']='user'
    output.mkdir(parents=True)
    write_lines(output/'candidates.jsonl',rows)
    write(output/'answers.json',read(answers_path))
    confirmed=sum(r.get('review',{}).get('human_reviewed') is True and r['score'] is not None for r in rows)
    write(output/'manifest.json',{'purpose':'Explicit user answers on teacher candidates; unanswered proposals unchanged',
          'created_date':created_date(),'policy':POLICY,'training_ready':False,'rows':len(rows),
          'user_confirmed_rows':confirmed,'teacher_proposed_rows':sum(r['review_status']=='teacher_proposed_unreviewed' for r in rows),
          'user_held_rows':sum(r['review_status']=='user_held_teacher_example' for r in rows),
          'parent_manifest_sha256':digest(source/'manifest.json'),'packet_sha256':digest(review_folder/'packet.json'),
          'files':{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    return rows


def apply_assistant_review(source, decisions_path, output):
    source,decisions_path,output=map(Path,(source,decisions_path,output))
    if output.exists():raise ValueError('preserve previous assistant review')
    verified_artifact(source)
    rows=load_lines(source/'candidates.jsonl');decisions=load_lines(decisions_path)
    by_id={d['id']:d for d in decisions}
    if len(by_id)!=len(decisions) or not by_id or not set(by_id)<={r['id'] for r in rows}:
        raise ValueError('unique known decision IDs required')
    for row in rows:
        decision=by_id.get(row['id'])
        if decision is None:continue
        if row.get('review',{}).get('human_reviewed') is True:
            raise ValueError('assistant cannot replace user-confirmed examples')
        if decision.get('reviewer')!='assistant' or decision.get('human_reviewed') is not False or not decision.get('reason'):
            raise ValueError('explicit assistant attribution and rationale required')
        if decision['input_sha256']!=input_key(row):raise ValueError('review input changed')
        score=decision['proposed_score']
        if score is not None and (type(score) is not int or score not in (1,2,3)):raise ValueError('invalid score')
        if targetless(row['state']['notification']['body']) and score not in (None,1):raise ValueError('reaction policy conflict')
        row['score']=score;row['review']=deepcopy(decision)
        row['review_status']='assistant_reviewed_teacher_example' if score is not None else 'assistant_held_teacher_example'
        row['experiment']['label_source']='assistant'
    audit=audit_rows([r for r in rows if r['score'] is not None])
    if audit['exact_input_label_conflicts'] or audit['content_window_score_warnings']:raise ValueError('review introduced conflicts')
    output.mkdir(parents=True)
    write_lines(output/'candidates.jsonl',rows);write_lines(output/'assistant-decisions.jsonl',decisions)
    write(output/'manifest.json',{'purpose':'Assistant semantic review; no automatic human certification',
        'created_date':created_date(),'policy':POLICY,'training_ready':False,'rows':len(rows),
        'user_confirmed_rows':sum(r.get('review',{}).get('human_reviewed') is True for r in rows),
        'assistant_reviewed_rows':sum(r['review_status']=='assistant_reviewed_teacher_example' for r in rows),
        'assistant_held_rows':sum(r['review_status']=='assistant_held_teacher_example' for r in rows),
        'parent_manifest_sha256':digest(source/'manifest.json'),'decisions_sha256':digest(decisions_path),
        'files':{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    return rows


def prepare_heldout_review(source, output):
    from filtering_training.review.prepare_real_review_packets import build_packet, write_packet
    source,output=Path(source),Path(output)
    verify_frozen(source)
    all_rows=load_lines(source/'development-heldout.jsonl')
    rows=[deepcopy(r) for r in all_rows if not r.get('review',{}).get('human_reviewed')]
    if not rows:raise ValueError('all heldout rows already individually confirmed')
    for row in rows:
        row['score']=None
        row['review']={'input_sha256':input_key(row)}
    cases=[{'case_id':chr(65+i) if i<7 else f'CSV{r["source_number"]}',
            'round':1 if i<7 else 2+(i-7)//14,'source_numbers':[r['source_number']],
            'title':'실제 CSV 원본 '+r['source_number'],
            'question':'현재 작업과의 관련도를 1·2·3점 또는 보류로 골라 줘.',
            'ambiguity':'알림에 없는 대화·화면 내용을 추측하지 않아도 돼.'} for i,r in enumerate(rows)]
    packet=build_packet(rows,{'cases':cases,'blind':True,'quality_labels':{},'previously_held_reason':'__none__'})
    packet.pop('packet_id')
    packet.update(overview=f'합성 교사 초안이 아닌 실제 CSV 평가용 원본이다. 전체 {len(all_rows)}건 중 이미 사용자 확정한 {len(all_rows)-len(rows)}건은 재질문하지 않고 나머지 {len(rows)}건만 표시한다.',
        instructions='앞서 확인한 교사 A~G와 다른 실제 CSV 원본이다. 우선 이번 1차 A~G부터 점수 또는 보류를 선택해 줘. 기존 모델·AI 제안 점수는 숨겼다.',
        source_proposals_sha256=digest(source/'development-heldout.jsonl'))
    packet['packet_id']=key(packet)
    return write_packet(output,packet,source_proposals_sha256=digest(source/'development-heldout.jsonl'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher-candidates',type=Path)
    p.add_argument('--review-teacher',action='store_true')
    p.add_argument('--review-first',type=int,nargs='+')
    p.add_argument('--answers',type=Path)
    p.add_argument('--review-folder',type=Path)
    p.add_argument('--assistant-decisions',type=Path)
    p.add_argument('--review-heldout',action='store_true')
    p.add_argument('--merge-teacher',type=Path)
    p.add_argument('--student',action='store_true')
    p.add_argument('--reference',type=Path)
    p.add_argument('--teacher',type=Path)
    p.add_argument('--real-review',type=Path)
    p.add_argument('--policy-decisions',type=Path)
    a=p.parse_args()
    if a.student:
        if not all((a.reference,a.teacher,a.real_review)):p.error('--student requires --reference, --teacher and --real-review')
        prepare_student(a.reference,a.source,a.teacher,a.real_review,a.output,a.policy_decisions)
    elif a.merge_teacher:merge_teacher(a.source,a.merge_teacher,a.output)
    elif a.review_heldout:prepare_heldout_review(a.source,a.output)
    elif a.assistant_decisions:apply_assistant_review(a.source,a.assistant_decisions,a.output)
    elif a.answers:
        if not a.review_folder:p.error('--review-folder required with --answers')
        apply_teacher_answers(a.source,a.review_folder,a.answers,a.output)
    elif a.review_teacher:prepare_review(a.source,a.output,first=a.review_first)
    elif a.teacher_candidates:import_teacher(a.source,a.teacher_candidates,a.output)
    else:freeze(a.source,a.output)
