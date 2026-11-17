"""CPU contracts: valid paper recipes, strict input errors, and true resume identity."""
import json
from pathlib import Path
import tempfile
import unittest

from mbt.config import ModelConfig
from mbt.training_config import DataConfig, Recipe, TrainingConfig, safe_name
from mbt.train import build_manifest, parser

ROOT = Path(__file__).resolve().parents[1]


class RecipeTests(unittest.TestCase):
    def test_paper_presets_and_exact_data_horizons(self):
        horizons = {'w1': (53110, 128), 'w2': (78370, 128), 'w3': (55258, 256), 'w4': (75189, 256)}
        recipes = sorted((ROOT / 'configs').glob('w*/*.json'))
        self.assertEqual(len(recipes), 44)
        names = set()
        for path in recipes:
            with self.subTest(path=path.name, scale=path.parent.name):
                recipe = Recipe.load(path)
                self.assertNotIn(recipe.name, names)
                names.add(recipe.name)
                self.assertEqual((recipe.training.train_steps, recipe.training.global_batch_size), horizons[path.parent.name])
                self.assertEqual(recipe.training.sequence_length, 4096)
                self.assertEqual(recipe.training.eval_global_batch_size * recipe.training.eval_steps, 12160)
                self.assertIn(recipe.training.accumulation_steps(8), (1, 2))
                args = parser().parse_args(['--config', str(path), '--data-config', str(ROOT/'configs/data.example.json'),
                                           '--output-dir', '/scratch/mbt-unit-dry', '--world-size', '8',
                                           '--expert-parallel-size', '8', '--dry-run'])
                manifest = build_manifest(args)
                native = manifest['native_arguments']
                flags = [v for v in native if v.startswith('--')]
                self.assertEqual(len(flags), len(set(flags)))
                self.assertNotIn('--no-save-optim', flags)
                self.assertNotIn('--no-save-rng', flags)
                self.assertNotIn('--finetune', flags)
                self.assertEqual(native[native.index('--lr-decay-iters')+1], str(recipe.training.train_steps))

    def test_joint_shares_one_cache_per_depth_without_moving_depth_boundaries(self):
        independent = Recipe.load(ROOT/'configs/w4/balanced-4-independent.json').model
        joint = Recipe.load(ROOT/'configs/w4/balanced-4-joint.json').model
        self.assertEqual(joint.depth, independent.depth)
        self.assertEqual(joint.physical_layers, independent.physical_layers)
        # BF16 GQA2: K+V occupy 2*d bytes per token at each depth.
        self.assertEqual(joint.counts()['cache_bytes_per_token_bf16'], 2*(9*1408+6*768))
        self.assertEqual(independent.counts()['cache_bytes_per_token_bf16'], 2*(9*1408+24*768))
        self.assertEqual(independent.counts()['physical']-joint.counts()['physical'], 6*3*64)

    def test_scientific_config_typos_and_invalid_layouts_fail(self):
        with self.assertRaises(ValueError): ModelConfig.from_dict({'scale':'W1', 'branches_count':2})
        with self.assertRaises(ValueError): ModelConfig(scale='W1', family='wide-shallow',prelude=3,branches=(2,3),coda=3,kv_mode='joint')
        with self.assertRaises(ValueError): ModelConfig(scale='W1', family='standard',kv_mode='joint')
        with self.assertRaises(ValueError): TrainingConfig(learning_rate=0,train_steps=10,global_batch_size=128)
        with self.assertRaises(ValueError): TrainingConfig(learning_rate=.001,train_steps=10,global_batch_size=128,micro_batch_size=24).accumulation_steps(8)
        with self.assertRaises(ValueError): TrainingConfig.from_dict(dict(learning_rate=.001,train_steps=10,global_batch_size=128,lr_decay_iter=100))

    def test_names_include_final_suffix_and_enforce_limit(self):
        safe_name('x'*120)
        for name in ('x'*121, 'my run', '实验', 'foo/bar', ''):
            with self.assertRaises(ValueError): safe_name(name)
        args = parser().parse_args(['--config',str(ROOT/'tests/native_smoke.json'),'--output-dir','/scratch/mbt-test','--smoke-steps','4'])
        first = build_manifest(args)
        self.assertEqual(first['schema'], 2)
        self.assertEqual(first['model_format'], 'mbt-hierarchical-v2')
        args.resume = True
        args.stop_after_step = 2
        resumed = build_manifest(args)
        self.assertEqual(first['identity_sha256'], resumed['identity_sha256'])
        args.learning_rate = .001
        self.assertNotEqual(first['identity_sha256'], build_manifest(args)['identity_sha256'])
        self.assertTrue(first['name'].endswith('-smoke'))

    def test_data_contract_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.json'
            path.write_text(json.dumps({'train':[{'weight':-1,'prefix':'/tmp/train'}],
                                       'validation':[{'weight':1,'prefix':'/tmp/valid'}],
                                       'tokenizer_model':'/tmp/tokenizer.model'}))
            with self.assertRaises(ValueError): DataConfig.load(path)
        data = DataConfig.load(ROOT/'configs/data.example.json')
        self.assertEqual(data.backend_args()[-2:], ['--tiktoken-num-special-tokens','256'])
        with self.assertRaises(ValueError): data.validate_files()

    def test_dry_run_does_not_import_gpu_libraries(self):
        import os
        import subprocess
        import sys
        # Other tests may intentionally import the explicit decoder in this
        # process. The public dry-run contract belongs to a fresh process.
        code = '''
import sys
from pathlib import Path
from mbt.train import parser, build_manifest
root = Path(sys.argv[1])
args = parser().parse_args([
    '--config', str(root/'configs/w1/balanced-2-joint.json'),
    '--data-config', str(root/'configs/data.example.json'),
    '--output-dir', '/scratch/mbt-dry-import-test',
    '--world-size', '8', '--expert-parallel-size', '8', '--dry-run'])
build_manifest(args)
unexpected = [name for name in ('torch', 'megatron', 'transformer_engine') if name in sys.modules]
if unexpected:
    raise AssertionError('dry run imported GPU libraries: '+', '.join(unexpected))
print('PASS')
'''
        env = dict(os.environ, PYTHONPATH=str(ROOT/'code'), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-c', code, str(ROOT)], env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
