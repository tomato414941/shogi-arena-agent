import unittest

import shogi_arena_agent


class PackageTest(unittest.TestCase):
    def test_version_exists(self) -> None:
        self.assertEqual(shogi_arena_agent.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
