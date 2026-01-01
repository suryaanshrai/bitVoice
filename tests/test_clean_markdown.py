
import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bitvoice import clean_markdown

class TestCleanMarkdown(unittest.TestCase):
    def test_frontmatter_removal(self):
        content = """---
title: Test Document
author: Me
---
# Actual Content
This is the content."""
        cleaned = clean_markdown(content)
        self.assertNotIn("title: Test Document", cleaned)
        self.assertNotIn("---", cleaned)
        self.assertIn("Actual Content", cleaned)

    def test_filename_insertion(self):
        content = "Some content."
        filename = "MyFile.md"
        cleaned = clean_markdown(content, filename=filename)
        # Should be at the start
        # Should start with header MyFile (header syntax removed)
        self.assertTrue(cleaned.startswith("MyFile"))
        self.assertIn("Some content", cleaned)

    def test_filename_and_frontmatter(self):
        content = """---
meta: data
---
Body text."""
        cleaned = clean_markdown(content, filename="Doc.md")
        self.assertNotIn("meta: data", cleaned)
        self.assertTrue(cleaned.startswith("Doc"))
        self.assertNotIn("Doc.md", cleaned)
        self.assertIn("Body text", cleaned)

if __name__ == "__main__":
    unittest.main()
