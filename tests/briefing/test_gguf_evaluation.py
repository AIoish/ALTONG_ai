import unittest
from unittest.mock import Mock, patch

from briefing_training.evaluate_gguf import GGUFGenerator, GGUFQwenBackend, validate_server_url


class GGUFEvaluationTests(unittest.TestCase):
    def test_pipeline_backend_preserves_messages_and_non_thinking_template(self):
        backend = object.__new__(GGUFQwenBackend)
        backend.tokenizer = Mock()
        backend.tokenizer.apply_chat_template.return_value = "pipeline prompt"
        backend.generator = Mock()
        backend.generator.complete.return_value = ('{"primary_category":"기타","summary_lines":[]}', 1.0)
        messages = [{"role": "user", "content": "source notifications"}]
        self.assertIn("primary_category", backend.generate(messages))
        backend.tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        backend.generator.complete.assert_called_once_with(backend.tokenizer, "pipeline prompt")

    def test_validation_only_uses_local_server(self):
        self.assertEqual(validate_server_url("http://127.0.0.1:18089/"), "http://127.0.0.1:18089")
        for url in ("https://example.com", "http://10.0.0.1:8080", "http://localhost/path",
                    "http://user:password@localhost:8080"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_server_url(url)

    def test_exact_tokens_greedy_no_grammar_and_no_cache(self):
        class Tokenizer:
            def encode(self, prompt, add_special_tokens):
                self.assert_prompt = prompt
                return [1, 2, 3]
        with patch("briefing_training.evaluate_gguf.render_generation_prompt", return_value="exact prompt"), \
             patch("briefing_training.evaluate_gguf.request_json", side_effect=[
                 {"tokens": [1, 2, 3]}, {"content": ' {"primary_category":"기타","summary_lines":["요약"]} '},
             ]) as request:
            content, elapsed = GGUFGenerator("http://localhost:18089")(
                tokenizer=Tokenizer(), model=None, group={}, do_sample=False, task="briefing")
        payload = request.call_args.args[1]
        self.assertEqual(payload["prompt"], [1, 2, 3])
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["repeat_penalty"], 1.0)
        self.assertFalse(payload["cache_prompt"])
        self.assertNotIn("grammar", payload)
        self.assertNotIn("json_schema", payload)
        self.assertTrue(content.startswith("{"))
        self.assertGreaterEqual(elapsed, 0)

    def test_token_mismatch_stops_before_inference(self):
        class Tokenizer:
            def encode(self, prompt, add_special_tokens):
                return [1]
        with patch("briefing_training.evaluate_gguf.render_generation_prompt", return_value="prompt"), \
             patch("briefing_training.evaluate_gguf.request_json", return_value={"tokens": [2]}) as request, \
             self.assertRaisesRegex(ValueError, "tokenization differ"):
            GGUFGenerator("http://localhost:18089")(tokenizer=Tokenizer(), model=None, group={})
        self.assertEqual(request.call_count, 1)
