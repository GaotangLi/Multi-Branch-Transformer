"""Retention must preserve foreign/incomplete checkpoints and the latest full save."""
import json
from pathlib import Path
import tempfile
import unittest

from mbt.runtime import record_and_prune


def fake_save(root, step):
    folder = root/f'iter_{step:07d}'
    folder.mkdir()
    for name in ('metadata.json', '.metadata', 'common.pt', '__0_0.distcp'):
        (folder/name).write_bytes(b'test fixture; not a real model')
    (root/'latest_checkpointed_iteration.txt').write_text(str(step))
    return folder


class RetentionTests(unittest.TestCase):
    def test_only_owned_complete_unchanged_saves_are_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            first = fake_save(root, 10)
            record_and_prune(root, 10, 'own-run', 1)
            foreign = fake_save(root, 11)
            record_and_prune(root, 11, 'another-run', 1)
            partial = root/'iter_0000012'; partial.mkdir()
            latest = fake_save(root, 20)
            report = record_and_prune(root, 20, 'own-run', 1)
            self.assertEqual(report['removed'], [10])
            self.assertFalse(first.exists())
            self.assertTrue(latest.exists())
            self.assertTrue(foreign.exists())
            self.assertTrue(partial.exists())

    def test_modified_old_checkpoint_stops_before_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            first = fake_save(root, 10)
            record_and_prune(root, 10, 'run', 1)
            (first/'common.pt').write_bytes(b'changed by another actor')
            latest = fake_save(root, 20)
            with self.assertRaises(ValueError): record_and_prune(root, 20, 'run', 1)
            self.assertTrue(first.exists())
            self.assertTrue(latest.exists())

    def test_symlink_or_wrong_tracker_never_prunes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            first = fake_save(root, 10)
            (root/'latest_checkpointed_iteration.txt').write_text('9')
            with self.assertRaises(ValueError): record_and_prune(root, 10, 'run', 1)
            (root/'latest_checkpointed_iteration.txt').write_text('10')
            (root/'foreign-link').symlink_to(first, target_is_directory=True)
            with self.assertRaises(ValueError): record_and_prune(root, 10, 'run', 1)
            self.assertTrue(first.exists())


if __name__ == '__main__':
    unittest.main()
