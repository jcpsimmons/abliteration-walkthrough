import json
from pathlib import Path
import tempfile
import unittest

from abliteration.data import partition, read_snapshot
from abliteration.chat import chat_messages

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_complete_disjoint_and_unseen(self):
        rows = [{'id': str(i)} for i in range(520)]
        seen = {str(i) for i in range(92)}
        parts = partition(rows, 104, 104, 42, seen)
        ids = {k: {r['id'] for r in v} for k, v in parts.items()}
        self.assertEqual({k: len(v) for k, v in ids.items()}, {'test':104, 'validation':104, 'candidates':312})
        self.assertFalse(ids['test'] & seen)
        self.assertFalse(ids['test'] & ids['validation'])
        self.assertFalse(ids['test'] & ids['candidates'])
        self.assertFalse(ids['validation'] & ids['candidates'])
        self.assertEqual(set.union(*ids.values()), {r['id'] for r in rows})
        self.assertEqual(parts, partition(rows, 104, 104, 42, seen))
        with self.assertRaisesRegex(ValueError, 'unseen'):
            partition(rows, 104, 104, 42, {r['id'] for r in rows})

    def test_snapshots_and_tamper_detection(self):
        cfg = json.loads((ROOT / 'config.json').read_text())
        for label, expected in [('harmful', 520), ('harmless', 31323)]:
            rows, meta = read_snapshot(ROOT / 'datasets', label, cfg)
            self.assertEqual(len(rows), expected)
            self.assertEqual(len({r['text'] for r in rows}), expected)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / 'manifest.json').write_bytes((ROOT / 'datasets/manifest.json').read_bytes())
            (tmp / meta['file']).write_text('modified')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_snapshot(tmp, 'harmless', cfg)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m['content']) for m in messages)))


class ChatTests(unittest.TestCase):
    def test_trims_complete_turns(self):
        history = [{'role':'user', 'content':'old'}, {'role':'assistant', 'content':'answer'},
                   {'role':'user', 'content':'new'}, {'role':'assistant', 'content':'reply'}]
        messages, ids = chat_messages(FakeTokenizer(), history, 'now', 'system', 17)
        self.assertEqual([m['content'] for m in messages], ['system','new','reply','now'])
        self.assertEqual(len(history), 4)
        with self.assertRaisesRegex(ValueError, 'budget'):
            chat_messages(FakeTokenizer(), [], 'too long', 'system', 5)


if __name__ == '__main__':
    unittest.main()
