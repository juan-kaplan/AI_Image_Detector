import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestPlaceholder(unittest.TestCase):
    def test_import(self):
        """Simple test to verify we can import from src."""
        try:
            import src.pipeline
            import src.models
        except ImportError:
            self.fail("Could not import src modules")

if __name__ == '__main__':
    unittest.main()
