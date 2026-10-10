"""Coverage, source isolation and alignment regressions for the new QA path."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from answer_planner import align_answer_sentences,evidence_plan,question_facets
from context_windows import expand_context_windows
from grounded_generation import split_evidence_sentences,validate_plain_answer,load_answer_generation_config
from grounded_generation import _answer_sources
from qna_pipeline import GroundedQAPipeline
from qna_fixtures import EVIDENCE
from unittest.mock import patch
import copy
import re


class EvidencePlanningTests(unittest.TestCase):
    def test_measurement_sentence_keeps_decimal_coefficient(self):
        text='The observed correlation coefficient is R=0.86 with damping 0.01, but the iterative solver does not converge.'
        ev={'evidence_id':'ev-01-a','page':3,'section':'Results','text':text}
        selected=split_evidence_sentences([ev])
        self.assertEqual(len(selected),1)
        result=validate_plain_answer(text,'What correlation and damping are reported?',selected,'en',load_answer_generation_config())
        self.assertIn('R=0.86',result['answer'])

    def test_standalone_equations_remain_ineligible(self):
        ev={'evidence_id':'ev-a','page':3,'section':'Results','text':'The equation is x = y + z and the system is therefore defined by this equation.'}
        self.assertEqual(split_evidence_sentences([ev]),[])

    def test_original_publisher_measurements_remain_eligible(self):
        correlation='Even in this difficult setting, the predicted and actual changes in loss were highly correlated (Pearson’s R = 0.86, Fig 2-Right).'
        damping='The model was not converged and the Hessian was not positive definite, so we added a damping term with λ = 0.01.'
        rows=split_evidence_sentences([{'evidence_id':'ev-01-a','page':5,'section':'Experiments','text':damping+' '+correlation}])
        self.assertTrue(any('0.86' in row['text'] for row in rows))
        self.assertTrue(any('λ = 0.01' in row['text'] for row in rows))
        result=validate_plain_answer(damping+' '+correlation,'What correlation and damping are reported?',rows,'en',load_answer_generation_config())
        self.assertIn('0.86',result['answer'])

    def test_cross_chunk_window_does_not_cross_section_or_paper(self):
        base={'paper_id':'p1','page':2,'section':'Methods','section_index':1}
        chunks=[{**base,'chunk_id':'a','chunk_index':1,'text':'The method uses these transforms: random cropping and color distortion.'},
                {**base,'chunk_id':'b','chunk_index':2,'text':'The transforms additionally include horizontal flipping and Gaussian blur.'},
                {**base,'chunk_id':'c','section_index':2,'section':'Results','chunk_index':3,'text':'Other section content.'},
                {**base,'paper_id':'p2','chunk_id':'d','chunk_index':3,'text':'Other paper content.'}]
        ev={**chunks[0],'evidence_id':'ev-a','locator':{'chunk_id':'a','page_label':'p. 2'}}
        expanded,origin=expand_context_windows([ev],chunks,'p1')
        self.assertIn('horizontal flipping',expanded[0]['text'])
        self.assertNotIn('Other',expanded[0]['text'])
        self.assertEqual(origin[0]['source_chunk_ids'],['a','b'])

    def test_window_must_not_silently_truncate_source(self):
        base={'paper_id':'p1','page':2,'section':'Methods','section_index':1}
        chunks=[{**base,'chunk_id':'a','chunk_index':1,'text':'a'*80},{**base,'chunk_id':'b','chunk_index':2,'text':'b'*80}]
        ev={**chunks[0],'evidence_id':'ev-a','locator':{'chunk_id':'a'}}
        expanded,origin=expand_context_windows([ev],chunks,'p1',max_chars=100)
        self.assertEqual(expanded[0]['text'],chunks[0]['text']);self.assertEqual(origin,[])

    def test_missing_facet_is_visible_and_not_semantic_proof(self):
        plan=evidence_plan('Which model is used and what accuracy is reported?',
             [{'sentence_id':'s1','evidence_id':'e1','requested_facet_ids':['request-1']}],True)
        self.assertEqual(plan['missing_facet_ids'],['request-2'])
        self.assertFalse(plan['facets'][0]['semantic_support_verified'])

    def test_alignment_retains_each_source_and_rejects_count_mismatch(self):
        source=[{'sentence_id':'s1','evidence_id':'e1','page':2,'text':'The method does not change the labels.'}]
        aligned=align_answer_sentences(source,['이 방법은 레이블을 변경하지 않습니다.'])
        self.assertEqual(aligned[0]['source_text'],source[0]['text'])
        self.assertEqual(aligned[0]['answer_language'],'ko')
        with self.assertRaises(ValueError):align_answer_sentences(source,[])

    def test_pronoun_continuation_retains_question_subject(self):
        facets=question_facets('Does weighted SVD remove bias, and what does it change?')
        self.assertIn('weighted SVD',facets[1]['query'])
        self.assertNotIn('weighted SVD',facets[1]['request'])

    def test_final_budget_never_removes_an_anchor_qualifier(self):
        config=load_answer_generation_config()
        config['evidence_planning']['enabled']=True
        config['answer']['maximum_characters']=110
        rows=[{'sentence_id':'s1','anchor_sentence_id':'s1','text':'The method improves accuracy on the reported dataset.'},
              {'sentence_id':'s2','anchor_sentence_id':'s1','text':'However, this result only applies to the controlled setting used in the study.'},
              {'sentence_id':'s3','anchor_sentence_id':'s3','text':'The study evaluated a separate baseline.'}]
        selected=_answer_sources('What results are reported?',rows,'en',config)
        self.assertFalse(any(s['sentence_id'] in {'s1','s2'} for s in selected))
        self.assertEqual([s['sentence_id'] for s in selected],['s3'])

    def test_bounded_retry_keeps_paper_scope_and_contract_identifiers(self):
        class Retrieval:
            def __init__(self):self.calls=[]
            def run(self,query,paper_id):
                self.calls.append((query,paper_id))
                ev=copy.deepcopy(EVIDENCE[0])
                if len(self.calls)>1:
                    ev['chunk_id']='paper-001-p003-c002'
                    ev['page']=3
                return {'evidence':[ev],'production':{'config_fingerprint':'fixture'}}
        retrieval=Retrieval()
        pipeline=GroundedQAPipeline(retrieval=retrieval,sentence_selector=object(),
            answer_strategy='evidence_first',enable_abstention=False)
        pipeline.answer_generation_config['evidence_planning'].update(enabled=True,require_all_facets=True,maximum_retrieval_retries=1)
        plan={'facets':[{'facet_id':'request-1','request':'Which method is used','query':'Which method is used'},
                {'facet_id':'request-2','request':'what accuracy is reported',
                'query':'Which method is used; what accuracy is reported'}], 'missing_facet_ids':['request-2']}
        generated={'response':{'answer':'The answer cannot be verified from the provided paper evidence.',
            'evidence_ids':[],'sufficiency':'insufficient','abstention_reason':'No source candidate'},
            'attempts':[{'status':'invalid','reason_code':'requested_facet_not_supported','attempt':0,'raw_response':''}],
            'fallback_used':False,'stage_timings':{},'evidence_plan':plan,'selected_sentences':[]}
        first=copy.deepcopy(generated)
        first['response']={'answer':'The paper studies evidence-grounded question answering.',
            'evidence_ids':[EVIDENCE[0]['evidence_id']],'sufficiency':'sufficient','abstention_reason':None}
        first['attempts']=[{'attempt':1,'status':'valid','raw_response':first['response']['answer']}]
        with patch('qna_pipeline.generate_evidence_first_answer',side_effect=[first,copy.deepcopy(generated)]) as generate:
            result=pipeline.ask('Which method is used and what accuracy is reported?','paper-001')
        self.assertEqual(len(retrieval.calls),2)
        retry_sources=generate.call_args.args[3]
        self.assertTrue(all(e['paper_id']=='paper-001' for e in retry_sources))
        self.assertTrue(all(re.fullmatch(r'ev-[0-9]{2}-.+',e['evidence_id']) for e in retry_sources))
        self.assertEqual(len({e['evidence_id'] for e in retry_sources}),len(retry_sources))
        self.assertFalse(result['pipeline']['evidence_context']['bounded_retry']['adopted'])
        self.assertEqual(result['response']['sufficiency'],'sufficient')
        self.assertEqual(result['response']['answer'],first['response']['answer'])


if __name__=='__main__':unittest.main()
