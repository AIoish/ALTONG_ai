"""용도: 동결 학습 비중·신규 시나리오 총비중·0% 대조군과 잘못된 설정 거부를 검사한다.
생성일: 2026-11-15
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from filtering_training.common.relevance_mixture import mixture_weights
from filtering_training.common.artifact_io import read, write
from filtering_training.modeling.relevance_only import configure_distillation
from filtering_training.modeling.run_relevance_mixture import study
from filtering_training.preparation.curate_relevance import write_lines


def sample(identifier, source, group):
    return {'id':identifier, 'score':1, 'split':'train', 'state':{'body':identifier}, 'experiment_group':group,
            'experiment':{'origin':source, 'loss_origin':'pilot' if source=='reviewed' else source}}


class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.reference = [sample('a','balanced','a'),sample('b','balanced','b'),
                          sample('p','pilot','p'),sample('r','reviewed','r'),sample('x','real','x')]
        self.rows = deepcopy(self.reference)+[dict(sample('s1','pilot','s'),training_only=True),
                                                dict(sample('s2','pilot','s'),training_only=True),
                                                dict(sample('t','pilot','t'),training_only=True)]

    def test_old_relative_mass_preserved_and_supplement_mass_explicit(self):
        zero, z = mixture_weights(self.rows,self.reference,0)
        low, l = mixture_weights(self.rows,self.reference,.05)
        self.assertEqual(z['zero_weight_rows'],3)
        self.assertAlmostEqual(sum(low)/len(low),1)
        self.assertAlmostEqual(l['groups']['supplement']['mass'],.05)
        for a,b in zip(zero[:5],low[:5]):self.assertAlmostEqual(b/a,.95)
        self.assertAlmostEqual(l['groups']['reviewed']['mass'],.95/6)

    def test_app_variants_do_not_increase_scenario_total(self):
        old,_=mixture_weights(self.rows,self.reference,.1)
        extra=deepcopy(self.rows)+[dict(sample('s3','pilot','s'),training_only=True)]
        new,_=mixture_weights(extra,self.reference,.1)
        self.assertAlmostEqual(sum(old[i] for i in (5,6))/len(old),sum(new[i] for i in (5,6,8))/len(new))
        self.assertAlmostEqual(old[7]/len(old),new[7]/len(new))

    def test_explicit_groups_and_invalid_reference_rejected(self):
        weights,report=mixture_weights(self.rows,self.reference,.1,base_shares={'balanced':.4,'pilot':.2,'reviewed':.1,'real':.3})
        self.assertAlmostEqual(report['groups']['balanced']['mass'],.36)
        for share in (-.1,1,float('nan')):
            with self.assertRaises(ValueError):mixture_weights(self.rows,self.reference,share)
        tampered=deepcopy(self.rows);tampered[0]['score']=2
        with self.assertRaises(ValueError):mixture_weights(tampered,self.reference,.1)
        with self.assertRaises(ValueError):mixture_weights(self.reference,self.reference,.1)


    def test_teacher_mass_uses_authored_scenarios_instead_of_lineage_groups(self):
        teachers = [sample('t1','teacher_distilled','shared-lineage'),
                    sample('t2','teacher_distilled','shared-lineage'),
                    sample('t3','teacher_distilled','shared-lineage')]
        for r, scenario in zip(teachers, ['one', 'one', 'two']):
            r['experiment']['train_only'] = True
            r['scenario'] = scenario
        rows = self.reference + teachers
        with tempfile.TemporaryDirectory() as temp:
            write(Path(temp)/'manifest.json', {'distillation_preparation': {'verified': True}})
            config = {'dataset_dir': temp}
            report = configure_distillation({'train': rows, 'validation': []}, config, .1)
        weights = config['row_loss_weights']
        self.assertEqual(report['groups']['supplement']['scenarios'], 2)
        self.assertAlmostEqual(weights[-1], weights[-2] + weights[-3])
        for weight in weights[:5]:self.assertAlmostEqual(weight, len(rows)*.9/5)
        self.assertAlmostEqual(sum(weights[-3:])/len(rows), .1)
        self.assertEqual({r['experiment_group'] for r in teachers}, {'shared-lineage'})
        teachers[0]['split'] = 'validation'
        with self.assertRaisesRegex(ValueError,'training-only'):
            mixture_weights(rows,self.reference,.1,scenario_field='scenario')

    def test_teacher_missing_scenario_or_permission_is_rejected(self):
        teacher = sample('t','teacher_distilled','lineage')
        with self.assertRaisesRegex(ValueError,'training-only'):
            mixture_weights(self.reference+[teacher],self.reference,.1,scenario_field='scenario')
        teacher['experiment']['train_only']=True
        with self.assertRaisesRegex(ValueError,'scenario'):
            mixture_weights(self.reference+[teacher],self.reference,.1,scenario_field='scenario')

    def test_study_pairs_fresh_controls_and_only_repeats_promising_share(self):
        teacher = sample('t','teacher_distilled','lineage')
        teacher.update(scenario='scenario')
        teacher['experiment']['train_only']=True
        validation = [sample('v1','balanced','v1'),sample('v2','balanced','v2'),
                      sample('v3','real_development','v3')]
        for promising in (False, True):
            with self.subTest(promising=promising), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                write(root/'manifest.json', {'distillation_preparation':{'verified':True}})
                rows={'train':self.reference+[teacher],'validation':validation}
                config={'dataset_dir':str(root),'code_hashes':{}}
                calls=[]
                def fake_train(folder, actual_rows, encoded, tokenizer, cfg, settings):
                    calls.append((cfg['training_mixture']['supplement_share'],settings['seed']))
                    self.assertIs(actual_rows,rows)
                    folder.mkdir()
                    scores=[1,1,1] if promising and calls[-1][0] else [2,1,1]
                    write_lines(folder/'validation.predictions.jsonl',
                                [{'id':r['id'],'predicted':s} for r,s in zip(validation,scores)])
                    write(folder/'run_config.json',{'checkpoint_sha256':'mock-checkpoint'})
                with patch('filtering_training.modeling.run_relevance_mixture.prepare',return_value=(rows,{},None,config)), \
                     patch('filtering_training.modeling.run_relevance_mixture.train',side_effect=fake_train):
                    result=study(root/'study',dataset=root,allow_candidates=True)
                self.assertEqual(len(calls),7 if promising else 3)
                self.assertEqual(result['selected_share'],.05 if promising else None)
                self.assertEqual(len(result['paired_decisions']),3 if promising else 0)
                self.assertFalse(result['deployment_approved'])
                self.assertEqual(read(root/'study/status.json')['runs'],len(calls))


if __name__ == '__main__':
    unittest.main()
