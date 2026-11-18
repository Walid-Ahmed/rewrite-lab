import unittest
from evaluate_joint_holdout import HOLDOUT, evaluate, load_holdout, summarize


@unittest.skipUnless((HOLDOUT / "manifest.json").exists(), "needs the private holdout data (set DATA_ROOT)")
class JointHoldoutTests(unittest.TestCase):
    def test_perfect_predictions_score_zero_error(self):
        rows = load_holdout()
        result = evaluate({r['holdout_id']: r['label_pct'] for r in rows}, rows)
        self.assertEqual(result['all']['n'], 150)
        self.assertEqual(result['all']['mae_pp'], 0)
        self.assertEqual(result['all']['auc_0_vs_100'], 1)
        self.assertEqual(set(result), {'ai_generated', 'human_likely_pre_2022', 'all'})

    def test_hash_keys_work_and_missing_rows_fail(self):
        rows = load_holdout()
        by_hash = {r['text_sha256']: 50.0 for r in rows}
        self.assertEqual(evaluate(by_hash, rows)['all']['n'], 150)
        by_hash.pop(rows[0]['text_sha256'])
        with self.assertRaises(SystemExit):
            evaluate(by_hash, rows)

    def test_summary_counts_confident_misses(self):
        s = summarize([10, 90, 50], [100, 0, 50])
        self.assertEqual(s['confident_misses'], 2)
        self.assertEqual(s['side_agreement'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
