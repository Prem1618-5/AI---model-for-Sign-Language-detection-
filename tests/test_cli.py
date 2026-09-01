"""
CLI and Argument Parsing Test Suite
Tests CLI argument parsing, help messages, subcommand configurations, and default arguments.
"""

import os
import sys
import subprocess
import unittest

# Ensure src path is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
MAIN_SCRIPT = os.path.join(SRC_DIR, 'main.py')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestCLIParsing(unittest.TestCase):
    """Tier 3: Verifies CLI argument parsing and help output."""

    def run_cli(self, args: list[str]) -> subprocess.CompletedProcess:
        """Helper to run main.py as a subprocess in the current python interpreter."""
        cmd = [sys.executable, MAIN_SCRIPT] + args
        return subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10
        )

    def test_main_help_flag(self):
        """Test that running `main.py --help` returns 0 and prints usage information."""
        result = self.run_cli(['--help'])
        self.assertEqual(result.returncode, 0, f"Error: {result.stderr}")
        self.assertIn("Sign Language Detection ML System", result.stdout)
        self.assertIn("collect", result.stdout)
        self.assertIn("preprocess", result.stdout)
        self.assertIn("train", result.stdout)
        self.assertIn("evaluate", result.stdout)
        self.assertIn("recognize", result.stdout)

    def test_no_arguments_prints_help(self):
        """Test that running without arguments displays help."""
        result = self.run_cli([])
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage: main.py", result.stdout)

    def test_collect_subcommand_help(self):
        """Test that `collect --help` displays expected argument options."""
        result = self.run_cli(['collect', '--help'])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--gestures", result.stdout)
        self.assertIn("--samples", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_collect_missing_required_gestures(self):
        """Test that `collect` fails with non-zero code when required --gestures is omitted."""
        result = self.run_cli(['collect'])
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(
            "the following arguments are required: --gestures" in result.stderr or
            "required" in result.stderr.lower()
        )

    def test_preprocess_subcommand_help(self):
        """Test that `preprocess --help` displays augmentation, input, and output flags."""
        result = self.run_cli(['preprocess', '--help'])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--augment", result.stdout)
        self.assertIn("--input", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_train_subcommand_help(self):
        """Test that `train --help` displays model-type, epochs, batch-size flags."""
        result = self.run_cli(['train', '--help'])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--model-type", result.stdout)
        self.assertIn("--epochs", result.stdout)
        self.assertIn("--batch-size", result.stdout)
        self.assertIn("--data", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_train_invalid_model_type(self):
        """Test that `train --model-type invalid_type` is rejected by argparse choices."""
        result = self.run_cli(['train', '--model-type', 'transformer_unsupported'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr.lower())

    def test_evaluate_subcommand_help(self):
        """Test that `evaluate --help` displays model path and data path options."""
        result = self.run_cli(['evaluate', '--help'])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--model", result.stdout)
        self.assertIn("--data", result.stdout)

    def test_recognize_subcommand_help(self):
        """Test that `recognize --help` displays camera, threshold, and no-flip options."""
        result = self.run_cli(['recognize', '--help'])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--model", result.stdout)
        self.assertIn("--camera", result.stdout)
        self.assertIn("--threshold", result.stdout)
        self.assertIn("--no-flip", result.stdout)

    def test_invalid_subcommand(self):
        """Test that passing an invalid subcommand returns non-zero error."""
        result = self.run_cli(['non_existent_command'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr.lower())


if __name__ == '__main__':
    unittest.main()
