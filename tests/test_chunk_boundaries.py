"""Protect complete source claims across production chunk boundaries."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ingest_papers import run_pipeline, split_text


class ChunkBoundaryTests(unittest.TestCase):
    def test_complete_numeric_claim_is_kept_in_one_chunk(self):
        claim = "The decoder requires 100 GB of memory and achieves 91.5% accuracy."
        text = "A separate introductory observation precedes the measurement. " + claim + " Further analysis follows."
        chunks = split_text(text, 100, 20)
        self.assertTrue(any(claim in chunk for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertTrue(any("100 GB" in chunk for chunk in chunks))

    def test_long_sentence_preserves_all_words_with_overlap(self):
        text = " ".join("measurement" + str(index) for index in range(80)) + "."
        chunks = split_text(text, 100, 20)
        observed = {word for chunk in chunks for word in chunk.split()}
        self.assertTrue(set(text.split()) <= observed)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(chunks, split_text(text, 100, 20))

    def test_no_overlap_preserves_nonwhitespace_source_characters(self):
        text = "The first result is 4 GB. The second result is 100 ms. " + "abcdefghij" * 20
        chunks = split_text(text, 65, 0)
        self.assertEqual("".join("".join(chunks).split()), "".join(text.split()))

    def test_invalid_windows_fail_instead_of_looping(self):
        for size, overlap in ((0, 0), (10, 10), (10, -1), (True, 0)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                split_text("A source fact.", size, overlap)

    def test_absent_pdf_inputs_preserve_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = [root / name for name in ("chunks.jsonl", "pages.jsonl", "report.json")]
            for output in outputs:
                output.write_bytes(b"existing verified artifact\r\n")
            empty = root / "empty"
            empty.mkdir()
            for papers in (empty, root / "missing"):
                with self.subTest(papers=papers), self.assertRaisesRegex(ValueError, "No paper"):
                    run_pipeline(papers, *outputs, 1200, 200)
                self.assertEqual([p.read_bytes() for p in outputs], [b"existing verified artifact\r\n"] * 3)


if __name__ == "__main__":
    unittest.main()
