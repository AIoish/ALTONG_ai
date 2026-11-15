"""용도: 동결 학습 비중·신규 시나리오 총비중·0% 대조군과 잘못된 설정 거부를 검사한다.
생성일: 2026-11-15
"""

from copy import deepcopy
import unittest

from filtering_training.common.relevance_mixture import mixture_weights


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


if __name__ == '__main__':
    unittest.main()
