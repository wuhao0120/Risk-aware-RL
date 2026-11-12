"""Budget, method selection, and launch isolation across frozen experiments."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

p = Path(__file__).resolve().parents[1] / 'tools/experiment.py'
spec = importlib.util.spec_from_file_location('launcher', p)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class StartupTests(unittest.TestCase):
    def test_frozen_sources(self):
        self.assertGreater(launcher.verify(), 1000)

    def test_train_budgets_and_cli_are_consistent(self):
        for key, config in launcher.REGISTRY.items():
            cmd, total, points = launcher.plan(key, 2, Path('/tmp/new-machine'), 'train')
            self.assertEqual(total, config['batches'] * config['batch'] * 1000)
            self.assertEqual(points[-1], total)
            if config['method'] == 'dca':
                self.assertEqual(int(cmd[cmd.index('--main-batches') + 1]) + 20, config['batches'])
                self.assertEqual(cmd[cmd.index('--arm') + 1], config['arm'])
            if config['method'] == 'tqpo' and not config.get('supports_batches'):
                self.assertNotIn('--batches', cmd)
            self.assertTrue(Path(cmd[2]).is_file())
            self.assertEqual(cmd[cmd.index('--seed') + 1], '2')

    def test_smoke_budget_preserves_dca_warmup(self):
        for key, config in launcher.REGISTRY.items():
            cmd, steps, points = launcher.plan(key, 71, Path('/tmp/smoke'), 'smoke')
            expected = {'dca': 22, 'qcpo': 10, 'tqpo': 2}[config['method']]
            self.assertEqual(steps, expected * config['batch'] * 1000)
            self.assertEqual(points, [steps])
            if config['method'] == 'dca':
                self.assertEqual(cmd[cmd.index('--main-batches') + 1], '2')

    def test_dry_run_creates_no_output(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'experiment'
            launcher.run('dynamic-dca_div200', 3, 0, output, 'train', dry_run=True)
            self.assertFalse(output.exists())

    def test_output_reuse_rejects_changed_seed(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            (output / 'launch.json').write_text('{}')
            with patch.object(launcher, 'verify', return_value=1248), self.assertRaisesRegex(RuntimeError, 'different experiment'):
                launcher.run('dynamic-dca_div25', 3, 0, output, 'train')


if __name__ == '__main__':
    unittest.main()
