"""Attribute residual transfer: explicit provenance, capacity, and isolation."""

import copy
import json
import unittest

import torch
from torch.nn import functional as F

from models.attribute_transfer import fit_attribute_transfer, complete_attribute_transfer


def fixture(classes=8, shots=12, dimension=12):
    rng = torch.Generator().manual_seed(702)
    centers = F.normalize(torch.randn(classes,dimension,generator=rng),dim=-1)
    labels = torch.arange(classes).repeat_interleave(shots)
    features = F.normalize(centers[labels]+.08*torch.randn(classes*shots,dimension,generator=rng),dim=-1)
    text = F.normalize(centers[:,None,:]+.07*torch.randn(classes,4,dimension,generator=rng),dim=-1)
    available = torch.ones(classes,4,dtype=torch.bool)
    masks = torch.zeros(classes,4,dimension,dtype=torch.bool)
    for g in range(4):
        masks[:,g,g*3:g*3+3] = True
    return features,labels,text,available,masks


def fitted(features,labels,text,available,classes=(0,1,2,3,4)):
    return fit_attribute_transfer(features,labels,text[list(classes)],available[list(classes)],list(classes),topk=3)


def predicted(state,features,labels,text,available,masks,classes=(5,6,7),**options):
    return complete_attribute_transfer(state,features,labels,text[list(classes)],available[list(classes)],
                                       masks[list(classes)],list(classes),**options)


class AttributeTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_bank_retains_only_fitting_statistics_and_exact_sampling_counts(self):
        x,y,text,available,_ = fixture()
        state = fitted(x,y,text,available)
        self.assertEqual(state['class_ids'].tolist(),list(range(5)))
        self.assertEqual(state['fit_sample_count'],60)
        self.assertEqual(state['within_degrees_of_freedom'],55)
        self.assertEqual(state['retained_training_samples'],0)
        self.assertNotIn('features',state)
        self.assertEqual(state['means'].shape,(5,12))
        for j in range(5):
            torch.testing.assert_close(state['means'][j],x[y==j].double().mean(0),atol=0,rtol=0)

    def test_unrequested_class_rows_do_not_change_any_fitting_statistic(self):
        x,y,text,available,_ = fixture()
        first = fitted(x,y,text,available)
        changed = x.clone()
        changed[y>=5] *= -50
        second = fitted(changed,y,text,available)
        for key in first:
            if torch.is_tensor(first[key]):
                self.assertTrue(torch.equal(first[key],second[key]),key)
            else:
                self.assertEqual(first[key],second[key],key)

    def test_modes_are_finite_and_diagnostics_are_json_serializable(self):
        x,y,text,available,masks = fixture()
        state = fitted(x,y,text,available)
        for mode in ('whole','semantic','random','shuffled'):
            for reliability in (False,True):
                result,diag = predicted(state,x,y,text,available,masks,mode=mode,
                                        reliability=reliability,return_diagnostics=True)
                self.assertEqual(result.shape,(3,12))
                self.assertEqual(result.dtype,x.dtype)
                self.assertTrue(torch.isfinite(result).all())
                json.dumps(diag,allow_nan=False)
                for cls in diag['classes']:
                    self.assertEqual(cls['support_count'],12)
                    self.assertTrue(all(0<=g['gate']<=1 for g in cls['groups']))
                    self.assertTrue(all(g['support_noise'] is None or g['support_noise']>=0 for g in cls['groups']))

    def test_unknown_attributes_and_empty_masks_are_exact_zero_corrections(self):
        x,y,text,available,masks = fixture()
        state = fitted(x,y,text,available)
        raw = torch.stack([x[y==c].mean(0) for c in (5,6,7)])
        missing = available.clone()
        missing[5:] = False
        for mode in ('whole','semantic','random','shuffled'):
            actual,diag = predicted(state,x,y,text,missing,masks,mode=mode,return_diagnostics=True)
            self.assertTrue(torch.equal(actual,raw))
            self.assertTrue(all(g['reason']=='unknown_attribute' for c in diag['classes'] for g in c['groups']))
            actual = predicted(state,x,y,text,available,torch.zeros_like(masks),mode=mode)
            self.assertTrue(torch.equal(actual,raw))

    def test_same_class_donor_is_excluded_and_no_donor_returns_exact_mean(self):
        x,y,text,available,masks = fixture()
        one = fitted(x,y,text,available,classes=(0,))
        actual,diag = predicted(one,x,y,text,available,masks,classes=(0,),return_diagnostics=True)
        self.assertTrue(torch.equal(actual[0],x[y==0].mean(0)))
        self.assertTrue(all(g['reason']=='no_eligible_donor' for g in diag['classes'][0]['groups']))
        state = fitted(x,y,text,available)
        _,diag = predicted(state,x,y,text,available,masks,classes=(0,),return_diagnostics=True)
        self.assertTrue(all(0 not in g['donor_ids'] for g in diag['classes'][0]['groups']))

    def test_future_support_and_unrequested_rows_do_not_change_current_completion(self):
        x,y,text,available,masks = fixture()
        state = fitted(x,y,text,available)
        for mode in ('semantic','random','shuffled'):
            first = predicted(state,x,y,text,available,masks,classes=(5,6),mode=mode,reliability=True)
            changed = x.clone()
            changed[y==6] *= -1
            changed[y==7] *= 4
            second = predicted(state,changed,y,text,available,masks,classes=(5,6),mode=mode,reliability=True)
            self.assertTrue(torch.equal(first[0],second[0]),mode)

    def test_random_masks_match_cardinality_and_are_independent_of_class_order(self):
        x,y,text,available,masks = fixture()
        state = fitted(x,y,text,available)
        first,diag = predicted(state,x,y,text,available,masks,mode='random',return_diagnostics=True)
        second,other = predicted(state,x,y,text,available,masks,classes=(7,5,6),mode='random',return_diagnostics=True)
        torch.testing.assert_close(first,second[[1,2,0]],atol=0,rtol=0)
        lookup={r['class_id']:r for r in other['classes']}
        for c in diag['classes']:
            self.assertEqual(c,lookup[c['class_id']])
            self.assertTrue(all(g['mask_count']==g['original_mask_count']==3 for g in c['groups']))

    def test_shuffling_deranges_only_current_available_masks(self):
        x,y,text,available,masks = fixture()
        state = fitted(x,y,text,available)
        available[5,1] = False
        _,diag = predicted(state,x,y,text,available,masks,classes=(5,),mode='shuffled',return_diagnostics=True)
        groups=diag['classes'][0]['groups']
        self.assertEqual([g['mask_source_group'] for g in groups],['color','part','pattern','shape'])
        self.assertEqual(groups[1]['reason'],'unknown_attribute')
        self.assertTrue(all(g['mask_count']==3 for g in groups))

    def test_exact_duplicate_attribute_evidence_is_merged(self):
        x,y,text,available,masks = fixture()
        text = text[:,:1].expand(-1,4,-1).clone()
        masks = masks[:,:1].expand(-1,4,-1).clone()
        single = available.clone()
        single[:,1:] = False
        first_state = fitted(x,y,text,single)
        repeated_state = fitted(x,y,text,available)
        for mode in ('whole','semantic','random','shuffled'):
            for reliability in (False,True):
                first = predicted(first_state,x,y,text,single,masks,mode=mode,reliability=reliability)
                repeated,diag = predicted(repeated_state,x,y,text,available,masks,mode=mode,
                                         reliability=reliability,return_diagnostics=True)
                self.assertTrue(torch.equal(first,repeated),mode)
                self.assertTrue(all(c['maximum_unique_mask_hits']<=1 for c in diag['classes']))
                if not reliability:
                    self.assertTrue(all(g['reason']=='duplicate_evidence_merged' for c in diag['classes'] for g in c['groups'][1:]))

    def test_overlap_is_averaged_and_never_changes_actual_support_count(self):
        x,y,text,available,masks = fixture()
        state = fitted(x,y,text,available)
        masks[:] = True
        _,diag = predicted(state,x,y,text,available,masks,return_diagnostics=True)
        for c in diag['classes']:
            self.assertEqual(c['support_count'],12)
            self.assertEqual(c['maximum_unique_mask_hits'],4)
            self.assertEqual(c['overlap_channels'],12)

    def test_donor_storage_order_and_similarity_ties_have_stable_global_identity_order(self):
        x,y,text,available,masks = fixture()
        text[:] = text[0,0].clone()
        first = fitted(x,y,text,available,classes=(0,1,2,3,4))
        second = fitted(x,y,text,available,classes=(4,2,0,3,1))
        a,da = predicted(first,x,y,text,available,masks,classes=(5,),return_diagnostics=True)
        b,db = predicted(second,x,y,text,available,masks,classes=(5,),return_diagnostics=True)
        self.assertTrue(torch.equal(a,b))
        for ga,gb in zip(da['classes'][0]['groups'],db['classes'][0]['groups']):
            self.assertEqual(ga['donor_ids'],gb['donor_ids'])
            self.assertEqual(ga['donor_ids'],[0,1,2])

    def test_invalid_shapes_nonfinite_values_and_settings_are_rejected(self):
        x,y,text,available,masks = fixture()
        for opts in ({'topk':0},{'topk':True},{'temperature':float('nan')},{'temperature':0}):
            with self.assertRaises(ValueError):
                fit_attribute_transfer(x,y,text[:5],available[:5],list(range(5)),**opts)
        with self.assertRaises(ValueError):
            fit_attribute_transfer(x,y,text[:5,:3],available[:5],list(range(5)))
        bad=x.clone(); bad[0,0]=float('nan')
        with self.assertRaises(ValueError):
            fitted(bad,y,text,available)
        state=fitted(x,y,text,available)
        with self.assertRaises(ValueError):
            predicted(state,x,y,text,available,masks,mode='future')
        with self.assertRaises(ValueError):
            predicted(state,x,y,text,available,masks.float()+.2)
        broken=copy.deepcopy(state); broken['within_diag'][0]=-1
        with self.assertRaises(ValueError):
            predicted(broken,x,y,text,available,masks)


if __name__=='__main__':
    unittest.main()
