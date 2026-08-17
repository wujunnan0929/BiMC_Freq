import os
import unittest
from unittest.mock import patch

from tools.generate_cub200_frequency_descriptions_deepseek import (
    DEFAULT_DEEPSEEK_MODEL,
    apply_deepseek_defaults,
    build_deepseek_request_body,
)


class DeepSeekFrequencyGeneratorTest(unittest.TestCase):
    def test_provider_defaults_are_isolated_and_use_max_reasoning(self):
        with patch.dict(os.environ, {}, clear=True):
            arguments = apply_deepseek_defaults(["--limit", "2"])

        self.assertIn("https://api.deepseek.com", arguments)
        self.assertIn(DEFAULT_DEEPSEEK_MODEL, arguments)
        self.assertIn("DEEPSEEK_API_KEY", arguments)
        effort_index = arguments.index("--reasoning-effort")
        self.assertEqual(arguments[effort_index + 1], "max")
        batch_index = arguments.index("--batch-size")
        token_index = arguments.index("--max-output-tokens")
        retry_index = arguments.index("--max-retries")
        self.assertEqual(arguments[batch_index + 1], "5")
        self.assertEqual(arguments[token_index + 1], "32768")
        self.assertEqual(arguments[retry_index + 1], "8")

    def test_explicit_command_line_model_is_not_overridden(self):
        with patch.dict(os.environ, {}, clear=True):
            arguments = apply_deepseek_defaults(
                ["--model", "deepseek-v4-flash", "--dry-run"]
            )

        self.assertEqual(arguments.count("--model"), 1)
        self.assertIn("deepseek-v4-flash", arguments)

    def test_thinking_request_uses_deepseek_parameters(self):
        body = build_deepseek_request_body(
            model="deepseek-v4-pro",
            user_prompt="Return JSON.",
            temperature=0.2,
            max_output_tokens=8192,
            reasoning_effort="max",
            json_mode=True,
        )

        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertEqual(body["max_tokens"], 8192)
        self.assertNotIn("temperature", body)
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_non_thinking_request_keeps_temperature(self):
        body = build_deepseek_request_body(
            model="deepseek-v4-flash",
            user_prompt="Return JSON.",
            temperature=0.2,
            max_output_tokens=4096,
            reasoning_effort="none",
            json_mode=True,
        )

        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["temperature"], 0.2)
        self.assertNotIn("reasoning_effort", body)


if __name__ == "__main__":
    unittest.main()
