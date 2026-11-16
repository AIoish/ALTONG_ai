"""용도: 교사 생성 전 묶음 동결·평가 출처 배제·초안 표기·원문 보존·검수 연결을 검증한다.
생성일: 2026-11-15
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from filtering_training.common.artifact_io import digest, load_lines, read, write
from filtering_training.preparation.curate_relevance import write_lines
from filtering_training.preparation.prepare_relevance_distillation import (
    apply_assistant_review, apply_teacher_answers, freeze, import_teacher, merge_teacher,
    prepare_heldout_review, prepare_review, prepare_student, student_materials, verify_frozen,
)
from filtering_training.preparation.prepare_relevance_experiment import input_key


def samples():
    return [{'id':f'raw-{s}-{i}','source_number':f'{s}-{i}','group_id':f'g-{s}-{i}','score':s,
        'state':{'notification':{'app_name':'Slack','sender':'담당','title':f't-{s}-{i}','body':f'body-{s}-{i}','timestamp':'2026-11-15T00:00:00Z'},
                 'context':{'active_process':'Code.exe','window_title':f'task-{s}-{i}','recent_windows':[]}},
        'review':{'human_reviewed':False},'provenance':{'received_kst':'2026-11-15 09:00:00','quality_flags':[]}} for s in (1,2,3) for i in range(10)]


class Tokenizer:
    def __call__(self,texts,**kwargs):
        return {'input_ids':[[1,2] for _ in texts],'attention_mask':[[1,1] for _ in texts]}


class DistillationTests(unittest.TestCase):
    def fixture(self,folder):
        path=folder/'source.jsonl';write_lines(path,samples())
        frozen=folder/'frozen';freeze(path,frozen)
        seed=load_lines(frozen/'generation-seeds.jsonl')[0]
        held=load_lines(frozen/'development-heldout.jsonl')[0]
        row={'id':'new','seed_ids':[seed['id']],'scenario':'new','case_kind':'useful_result','score':3,'split':'train',
             'state':deepcopy(seed['state']),'review_status':'teacher_proposed_unreviewed',
             'teacher':{'reviewer':'assistant','human_reviewed':False,'reason':'explicit task result'}}
        row['state']['notification']['body']='완료 결과 파일과 검증 로그를 전달했습니다.'
        return path,frozen,held,row

    def test_partition_is_frozen_before_generation_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);path,frozen,held,row=self.fixture(folder)
            self.assertEqual(len(load_lines(path)),30)
            seeds=load_lines(frozen/'generation-seeds.jsonl');evaluation=load_lines(frozen/'development-heldout.jsonl')
            self.assertFalse({r['experiment_group'] for r in seeds}&{r['experiment_group'] for r in evaluation})
            self.assertFalse(read(frozen/'manifest.json')['training_ready'])
            with self.assertRaises(ValueError):freeze(path,frozen)
            verify_frozen(frozen)
            write_lines(path,samples()[:-1])
            with self.assertRaisesRegex(ValueError,'source changed'):verify_frozen(frozen)

    def test_unknown_seed_heldout_body_policy_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);_,frozen,held,row=self.fixture(folder)
            invalid=[]
            r=deepcopy(row);r['seed_ids']=[held['id']];invalid.append([r])
            r=deepcopy(row);r['state']['notification']['body']=held['state']['notification']['body'];invalid.append([r])
            r=deepcopy(row);r['state']['notification']['body']='네';invalid.append([r])
            invalid.append([row,deepcopy(row)])
            for i,rows in enumerate(invalid):
                draft=folder/f'draft-{i}.jsonl';write_lines(draft,rows)
                with self.assertRaises(ValueError):import_teacher(frozen,draft,folder/f'out-{i}')

    def test_explicit_answers_only_confirm_selected_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);_,frozen,_,row=self.fixture(folder)
            other=deepcopy(row);other['id']='other';other['state']['notification']['app_name']='Discord'
            draft=folder/'draft.jsonl';write_lines(draft,[row,other])
            with patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                imported=import_teacher(frozen,draft,folder/'batch')
            prepare_review(folder/'batch',folder/'review')
            packet=read(folder/'review/packet.json')
            answers=folder/'answers.json';write(answers,{'packet_id':packet['packet_id'],'answers':{'A':{'score':3}}})
            rows=apply_teacher_answers(folder/'batch',folder/'review',answers,folder/'reviewed')
            self.assertTrue(rows[0]['review']['human_reviewed'])
            self.assertFalse(rows[0]['teacher']['human_reviewed'])
            self.assertEqual(rows[1],imported[1])
            self.assertEqual(rows[0]['state'],imported[0]['state'])
            self.assertFalse(read(folder/'reviewed/manifest.json')['training_ready'])
            self.assertEqual(read(folder/'reviewed/manifest.json')['user_confirmed_rows'],1)
            with self.assertRaises(ValueError):
                apply_teacher_answers(folder/'batch',folder/'review',answers,folder/'reviewed')

    def test_assistant_review_keeps_user_credit_and_rejects_gold_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);_,frozen,_,row=self.fixture(folder)
            other=deepcopy(row);other['id']='other';other['state']['notification']['app_name']='Discord'
            draft=folder/'draft.jsonl';write_lines(draft,[row,other])
            with patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                import_teacher(frozen,draft,folder/'batch')
            prepare_review(folder/'batch',folder/'cards')
            answers=folder/'answers.json';write(answers,{'packet_id':read(folder/'cards/packet.json')['packet_id'],'answers':{'A':{'score':3}}})
            before=apply_teacher_answers(folder/'batch',folder/'cards',answers,folder/'user')
            path=folder/'decisions.jsonl'
            decision={'id':before[0]['id'],'input_sha256':input_key(before[0]),'reviewer':'assistant','human_reviewed':False,'proposed_score':3,'reason':'manual review'}
            write_lines(path,[decision])
            with self.assertRaisesRegex(ValueError,'cannot replace'):
                apply_assistant_review(folder/'user',path,folder/'bad')
            decision.update(id=before[1]['id'],input_sha256=input_key(before[1]))
            write_lines(path,[decision])
            after=apply_assistant_review(folder/'user',path,folder/'assistant')
            self.assertEqual(before[0],after[0])
            self.assertFalse(after[1]['review']['human_reviewed'])
            self.assertEqual(read(folder/'assistant/manifest.json')['assistant_reviewed_rows'],1)

    def test_real_heldout_cards_hide_scores_and_reuse_existing_human_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);_,frozen,held,_=self.fixture(folder)
            originals=samples()
            next(r for r in originals if r['id']==held['id'])['review']['human_reviewed']=True
            source=folder/'source-with-human.jsonl';write_lines(source,originals)
            freeze(source,folder/'second-frozen')
            prepare_heldout_review(folder/'second-frozen',folder/'cards')
            packet=read(folder/'cards/packet.json')
            self.assertTrue(packet['blind'])
            self.assertTrue(all(c['suggested_score'] is None for c in packet['cases']))
            self.assertFalse(any(m['id']==held['id'] for c in packet['cases'] for m in c['members']))
            self.assertEqual(len(packet['cases']),len(load_lines(folder/'second-frozen/development-heldout.jsonl'))-1)

    def test_teacher_attribution_and_review_keep_proposals_unapproved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);_,frozen,_,row=self.fixture(folder)
            draft=folder/'draft.jsonl';write_lines(draft,[row]);before=digest(draft)
            with patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                imported=import_teacher(frozen,draft,folder/'batch')
            self.assertEqual(digest(draft),before)
            self.assertFalse(imported[0]['teacher']['human_reviewed'])
            self.assertFalse(read(folder/'batch/manifest.json')['training_ready'])
            prepare_review(folder/'batch',folder/'review',first=[1])
            packet=read(folder/'review/packet.json')
            self.assertEqual(packet['cases'][0]['members'][0]['id'],'new')
            self.assertEqual(packet['cases'][0]['suggested_score'],3)
            self.assertIn('교사 초안',packet['overview'])

    def student_fixture(self, folder):
        _,frozen,held,row=self.fixture(folder)
        draft=folder/'draft.jsonl';write_lines(draft,[row])
        with patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
            import_teacher(frozen,draft,folder/'teacher')
        evaluated=load_lines(frozen/'development-heldout.jsonl')
        for r in evaluated:
            r['review']={'reviewer':'user','human_reviewed':True,'proposed_score':r['score'],'input_sha256':input_key(r)}
        real=folder/'real';real.mkdir();write_lines(real/'candidates.jsonl',evaluated)
        write(real/'manifest.json',{'files':{'candidates.jsonl':digest(real/'candidates.jsonl')}})
        originals=load_lines(frozen/'generation-seeds.jsonl')+load_lines(frozen/'development-heldout.jsonl')
        for r in originals:
            r['split']='train';r['experiment']['origin']='real'
        # A different ID carrying a held-out input must be removed as well.
        alias=deepcopy(originals[0]);alias.update(id='held-alias',state=deepcopy(held['state']))
        alias['score']=held['score'];alias['experiment']={'origin':'balanced','train_only':False,'group_hints':['alias']}
        reference=folder/'reference';(reference/'relevance').mkdir(parents=True)
        write_lines(reference/'relevance/train.jsonl',originals+[alias])
        write_lines(reference/'relevance/validation.jsonl',[])
        write(reference/'manifest.json',{'files':{f'relevance/{s}.jsonl':{'sha256':digest(reference/f'relevance/{s}.jsonl')} for s in ('train','validation')}})
        return reference,frozen,folder/'teacher',real,held

    def test_student_moves_real_holdout_and_removes_indirect_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);reference,frozen,teacher,real,held=self.student_fixture(folder)
            before=digest(frozen/'development-heldout.jsonl')
            rows,excluded,near,changes,verification=student_materials(reference,frozen,teacher,real)
            train={r['id'] for r in rows if r['split']=='train'}
            evaluation={r['id'] for r in load_lines(real/'candidates.jsonl')}
            self.assertFalse(train&evaluation)
            self.assertNotIn('held-alias',train)
            self.assertTrue(evaluation<={r['id'] for r in rows if r['split']=='validation'})
            self.assertIn('held-alias',{r['id'] for r in excluded})
            self.assertEqual(before,digest(frozen/'development-heldout.jsonl'))
            self.assertEqual(verification['group_overlap'],0)

    def test_student_rejects_changed_real_input_or_unconfirmed_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);reference,frozen,teacher,real,_=self.student_fixture(folder)
            original=load_lines(real/'candidates.jsonl')
            for change in ('state','credit'):
                rows=deepcopy(original)
                if change=='state':rows[0]['state']['context']['window_title']='changed'
                else:rows[0]['review']['human_reviewed']=False
                write_lines(real/'candidates.jsonl',rows)
                write(real/'manifest.json',{'files':{'candidates.jsonl':digest(real/'candidates.jsonl')}})
                with self.assertRaisesRegex(ValueError,'human attribution changed'):
                    student_materials(reference,frozen,teacher,real)

    def test_student_policy_decisions_cannot_change_evaluation_or_human_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);reference,frozen,teacher,real,_=self.student_fixture(folder)
            path=folder/'policy.jsonl'
            for r in [load_lines(real/'candidates.jsonl')[0],load_lines(frozen/'generation-seeds.jsonl')[0]]:
                decision={'id':r['id'],'input_sha256':input_key(r),'proposed_score':1,'reviewer':'assistant','human_reviewed':False,'reason':'new policy'}
                write_lines(path,[decision])
                if r['split']=='validation':
                    with self.assertRaisesRegex(ValueError,'only to training'):
                        student_materials(reference,frozen,teacher,real,path)
                else:
                    result=student_materials(reference,frozen,teacher,real,path)
                    changed=next(x for x in result[0] if x['id']==r['id'])
                    self.assertEqual(changed['state'],r['state'])
                    self.assertFalse(changed['review']['human_reviewed'])
            # Protect a generation-side human row too, even with a valid assistant decision.
            seeds=load_lines(frozen/'generation-seeds.jsonl');seeds[0]['review']['human_reviewed']=True
            write_lines(frozen/'generation-seeds.jsonl',seeds)
            manifest=read(frozen/'manifest.json');manifest['files']['generation-seeds.jsonl']=digest(frozen/'generation-seeds.jsonl');write(frozen/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'replace human gold'):
                student_materials(reference,frozen,teacher,real,path)

    def test_student_verifier_rebuilds_metadata_even_if_output_hash_is_updated(self):
        from filtering_training.quality.verify_relevance_experiment import verify
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);reference,frozen,teacher,real,_=self.student_fixture(folder)
            with patch('filtering_training.quality.verify_relevance_experiment.verify',return_value={}):
                prepare_student(reference,frozen,teacher,real,folder/'student')
            rows=load_lines(folder/'student/relevance/train.jsonl');rows[0]['review_status']='falsely-certified'
            write_lines(folder/'student/relevance/train.jsonl',rows)
            manifest=read(folder/'student/manifest.json');manifest['files']['relevance/train.jsonl']['sha256']=digest(folder/'student/relevance/train.jsonl');write(folder/'student/manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'provenance changed'):
                verify(folder/'student',write_report=False)

    def test_merge_teacher_preserves_human_rows_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);_,frozen,_,row=self.fixture(folder)
            for name,identifier,app in [('first','one','Slack'),('second','two','Discord')]:
                r=deepcopy(row);r['id']=identifier;r['state']['notification']['app_name']=app
                path=folder/(name+'.jsonl');write_lines(path,[r])
                with patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                    import_teacher(frozen,path,folder/name)
            before=load_lines(folder/'first/candidates.jsonl')
            before[0]['review']={'reviewer':'user','human_reviewed':True,'proposed_score':3,'input_sha256':input_key(before[0])}
            write_lines(folder/'first/candidates.jsonl',before)
            manifest=read(folder/'first/manifest.json');manifest['files']['candidates.jsonl']=digest(folder/'first/candidates.jsonl');write(folder/'first/manifest.json',manifest)
            merged=merge_teacher(folder/'first',folder/'second',folder/'merged')
            self.assertEqual(merged[:1],before)
            with self.assertRaisesRegex(ValueError,'duplicate'):
                merge_teacher(folder/'first',folder/'first',folder/'duplicate')


if __name__=='__main__':
    unittest.main()
