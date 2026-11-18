import copy
import importlib.util
import unittest
from rewrite_quality_eval import (attach_reviews,digest,literal_checks,repetition,score_pair,summarize)


class PreservationScreeningTests(unittest.TestCase):
    def test_detects_quantities_identifiers_and_negation_without_claiming_accuracy(self):
        result=literal_checks('Clients must not retry within 30 seconds. Keep `item_id` and HTTP 429.',
                              'Clients can retry within 3 seconds. Keep `itemId` and HTTP 200.')
        kinds={f['kind'] for f in result['flags']}
        self.assertTrue({'missing_numbers','added_numbers','missing_technical_spans','negation_marker_count_changed'}<=kinds)
        self.assertIsNone(result['factual_accuracy_pct'])
    def test_missing_checkable_items_are_not_reported_as_perfect_preservation(self):
        result=literal_checks('Plants absorb water.','Plants take in water.')
        self.assertIsNone(result['number_token_retention']);self.assertIsNone(result['technical_span_retention'])
        self.assertEqual(result['status'],'screening_only_semantic_review_required')
    def test_identifier_substring_is_not_exact_preservation(self):
        self.assertIn('item_id',literal_checks('Use item_id.','Use other_item_id.')['missing_technical_spans'])
    def test_repetition_and_short_text_have_explicit_denominators(self):
        repeated=repetition('Trees shade the street. Trees shade the street.')
        self.assertGreater(repeated['repeated_ngram_excess_fraction'],0)
        self.assertEqual(repeated['duplicate_sentence_excess_count'],1)
        self.assertIsNone(repetition('Two words')['repeated_ngram_excess_fraction'])
    def test_semantic_review_requires_attribution_and_exact_output_evidence(self):
        row={'source_id':digest('A source.'),'model_label':'model','before':'A source.','after':'A rewrite.',
             'before_sha256':digest('A source.'),'after_sha256':digest('A rewrite.')}
        review={'source_id':row['source_id'],'model_label':'model','before_sha256':row['before_sha256'],'after_sha256':row['after_sha256'],
                'reviewer':'Codex','reviewer_type':'agent_llm','verdict':'preserved','claim_checks':[{'source_quote':'A source.','output_quote':'A rewrite.','status':'preserved'}]}
        result=attach_reviews([copy.deepcopy(row)],[review]);self.assertEqual(result[0]['semantic_review']['reviewer_type'],'agent_llm')
        for field in ('after_sha256','reviewer_type'):
            changed=copy.deepcopy(review);changed[field]='wrong'
            with self.assertRaises(ValueError):attach_reviews([copy.deepcopy(row)],[changed])
        changed=copy.deepcopy(review);changed['claim_checks'][0]['output_quote']='Invented evidence.'
        with self.assertRaises(ValueError):attach_reviews([copy.deepcopy(row)],[changed])


@unittest.skipUnless(importlib.util.find_spec('textstat'),'Run readability integration tests in .venv-quality')
class ReadabilityIntegrationTests(unittest.TestCase):
    def row(self,label,before='Trees provide shade. They cool streets.',after='Trees shade streets and help cool them.',finish='stop'):
        return score_pair({'before':before,'after':after,'model_label':label,'finish_reason':finish})
    def test_real_formulas_and_no_false_semantic_success(self):
        row=self.row('model');self.assertIsInstance(row['after_metrics']['readability']['flesch_reading_ease'],(float,int))
        self.assertEqual(row['semantic_review']['status'],'pending');self.assertIsNone(row['semantic_review']['factual_accuracy_pct'])
    def test_aggregate_uses_only_matched_complete_pairs(self):
        rows=[self.row('old'),self.row('new'),self.row('old','A second source.','A second rewrite.'),self.row('new','A second source.','Incomplete','length')]
        report=summarize(rows)
        self.assertEqual(report['old']['matched_comparison_samples'],1);self.assertEqual(report['new']['matched_comparison_samples'],1)
    def test_empty_output_is_incomplete_and_has_no_readability_score(self):
        for after in ('','...'):
            row=self.row('model',after=after);self.assertFalse(row['generation_complete']);self.assertIsNone(row['after_metrics']['readability']['flesch_reading_ease'])
        with self.assertRaisesRegex(ValueError,'lexical'):self.row('model',before='...')


if __name__=='__main__':unittest.main()
