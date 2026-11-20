"""Filters for the v8 code streams."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import prepare_acereason  # noqa: E402
import prepare_code  # noqa: E402


class CodeFilters(unittest.TestCase):
    def test_execution_result_is_kept_and_a_new_task_is_not(self):
        self.assertTrue(prepare_code.user_turn_is_execution("Execution result: \n16"))
        self.assertTrue(prepare_code.user_turn_is_execution(
            "Traceback (most recent call last):\nSyntaxError: invalid syntax"
        ))
        self.assertFalse(prepare_code.user_turn_is_execution(
            "Write a function that handles a SyntaxError in user input."
        ))
        self.assertFalse(prepare_code.user_turn_is_execution(
            "The code works well but lacks comments. Please add comments."
        ))

    def test_quotas_keep_subset_shares(self):
        counts = {"A": 50, "B": 30, "C": 20}
        quotas = prepare_code.subset_quotas(counts, 10)
        self.assertEqual(sum(quotas.values()), 10)
        self.assertEqual(quotas, {"A": 5, "B": 3, "C": 2})
        self.assertEqual(prepare_code.subset_quotas(counts, 200), counts)

    def test_kodcode_messages_drop_an_unclosed_think(self):
        closed = prepare_code.kodcode_messages([
            {"from": "human", "value": "Write a function."},
            {"from": "gpt", "value": "<think>\nstep\n</think>\n\ncode"},
        ])
        self.assertEqual(closed[0]["role"], "user")
        self.assertIn("</think>", closed[1]["content"])
        self.assertIsNone(prepare_code.kodcode_messages([
            {"from": "human", "value": "Write a function."},
            {"from": "gpt", "value": "<think>\nnever closed"},
        ]))

    def test_bird_user_turn_has_schema_and_question(self):
        schema = prepare_code.bird_schema("movie_platform", {
            "movie_platform|movies|title": "The title.",
            "movie_platform|movies|director": "Who directed it.",
            "other|t|c": "ignored",
        })
        text = prepare_code.bird_user(schema, "Who directed it?", "title = 'X'")
        self.assertIn("CREATE TABLE movies", text)
        self.assertIn("director", text)
        self.assertIn("Question: Who directed it?", text)
        self.assertIn("Evidence: title = 'X'", text)

    def test_arrow_files_skip_existing_samples(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "sft_data.parquet"))
            os.makedirs(os.path.join(root, "code_100k"))
            open(os.path.join(root, "sft_data.parquet", "a.parquet"), "w").close()
            open(os.path.join(root, "code_100k.parquet"), "w").close()
            open(os.path.join(root, "code_100k", "part-00.parquet"), "w").close()
            found = prepare_acereason.arrow_files(root)
            self.assertEqual(found, [os.path.join(root, "sft_data.parquet", "a.parquet")])


if __name__ == "__main__":
    unittest.main()
