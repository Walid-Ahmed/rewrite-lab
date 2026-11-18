import json
from pathlib import Path
import tempfile
import unittest
from ai_content_eval import FileDetector, checked_score, score_rows, summarize, write_report

class EvaluationTests(unittest.TestCase):
    def test_invalid_scores(self):
        for bad in [True,None,'50',-1,101,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): checked_score(bad)
        self.assertEqual(checked_score(0),0)

    def test_paired_denominator_and_direction(self):
        rows=[{'status':'scored','ai_content_before_pct':80,'ai_content_after_pct':20},
              {'status':'scored','ai_content_before_pct':0,'ai_content_after_pct':40},
              {'status':'detector_pending','ai_content_before_pct':100,'ai_content_after_pct':None}]
        s=summarize(rows)
        self.assertEqual(s['scored_pairs'],2)
        self.assertEqual(s['ai_content_before_pct'],40)
        self.assertEqual(s['ai_content_after_pct'],30)
        self.assertEqual(s['ai_content_reduction_pp'],10)
        self.assertEqual(s['worsened_fraction'],0.5)

    def test_exact_text_minimum_and_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scores.json'
            p.write_text(json.dumps({'detector_id':'test-fixture','minimum_words':2,
                'scores':[{'text':'before text','score_pct':75},{'text':'after text','score_pct':25}]}))
            detector=FileDetector(p)
            rows=[{'before':'before text','after':'after text','finish_reason':'stop'},
                  {'before':'short','after':'after text','finish_reason':'stop'},
                  {'before':'before text','after':'different text','finish_reason':'stop'},
                  {'before':'before text','after':'after text','finish_reason':'length'}]
            score_rows(rows,detector)
            self.assertEqual([r['status'] for r in rows],['scored','below_detector_word_minimum','detector_pending','generation_incomplete'])
            self.assertEqual(rows[0]['ai_content_reduction_pp'],50)
            self.assertIsNone(rows[3]['ai_content_after_pct'])
            self.assertEqual(summarize(rows)['scored_pairs'],1)

    def test_no_scores_are_not_zero(self):
        rows=score_rows([{'before':'x','after':'y','finish_reason':'stop'}],None)
        s=summarize(rows)
        self.assertNotIn('ai_content_after_pct',s)
        self.assertIsNone(rows[0]['ai_content_after_pct'])

    def test_report_and_wandb_payload(self):
        # Real wandb.Table construction, intercepted run: no network or real metrics.
        class Run:
            summary={}
            def log(self,payload): self.payload=payload
        run=Run()
        row={'id':'fixture','index':0,'before':'source','after':'rewrite','reference':'target',
             'status':'scored','ai_content_before_pct':80,'ai_content_after_pct':20,'ai_content_reduction_pp':60}
        with tempfile.TemporaryDirectory() as d:
            report=write_report([row],d,7,'test-fixture','final',run)
            self.assertEqual(report['summary']['ai_content_reduction_pp'],60)
            self.assertEqual(run.payload['iteration'],7)
            self.assertEqual(run.payload['validation_ai/final/ai_content_after_pct'],20)
            self.assertTrue((Path(d)/'report.md').exists())

class ConditionedPromptTests(unittest.TestCase):
    def test_validation_keeps_variant_prompts_and_distinct_ids(self):
        from ai_content_eval import validation_rows
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'valid.jsonl'
            rows=[{'messages':[{'role':'system','content':prompt},{'role':'user','content':'same original'},{'role':'assistant','content':'target'}]} for prompt in ['Academic instruction','Second instruction']]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows));loaded=validation_rows(folder)
            self.assertEqual([r['system_prompt'] for r in loaded],['Academic instruction','Second instruction'])
            self.assertNotEqual(loaded[0]['id'],loaded[1]['id']);self.assertEqual(loaded[0]['source_id'],loaded[1]['source_id'])

    def test_generation_uses_explicit_mode_view_instruction(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from ai_content_eval import generate_rewrite
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):self.messages=messages;return 'prompt'
        tokenizer=Tokenizer();fake=SimpleNamespace(stream_generate=lambda *a,**k:iter([SimpleNamespace(text='rewritten',finish_reason='stop')]))
        with patch.dict('sys.modules',{'mlx_lm':fake,'mlx_lm.sample_utils':SimpleNamespace(make_sampler=lambda **kw:None)}):
            text,finish=generate_rewrite(None,tokenizer,'source',10,system_prompt='Mode/view instruction')
        self.assertEqual(tokenizer.messages[0]['content'],'Mode/view instruction');self.assertEqual((text,finish),('rewritten','stop'))

if __name__=='__main__': unittest.main()
