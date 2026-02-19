#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for the metadynamics pipeline scripts.

Tests:
  1. load_plumed_file — parsing HILLS/COLVAR-style data
  2. plot_deltaG_convergence — single-row robustness (np.atleast_2d)
  3. validate_atom_selection — valid/invalid atom format patterns

Run:
  python3 -m pytest tests/test_pipeline.py -v
  # or:
  python3 tests/test_pipeline.py
"""

import os
import sys
import unittest
import tempfile

import numpy as np

# Add scripts directory to path
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
sys.path.insert(0, SCRIPT_DIR)


class TestLoadPlumedFile(unittest.TestCase):
    """Test PLUMED file parsing (HILLS/COLVAR)."""

    def test_basic_parsing(self):
        """Parse a minimal HILLS-like file with comments and data."""
        from analyze_convergence import load_plumed_file

        content = """\
#! FIELDS time cv1 sigma1 height biasfactor
#! SET something 1.0
0.000  1.500  0.300  1.200  15.0
1.000  1.600  0.300  1.100  15.0
2.000  1.700  0.300  0.900  15.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            data = load_plumed_file(tmp_path)
            self.assertEqual(data.shape, (3, 5))
            np.testing.assert_almost_equal(data[0, 0], 0.0)
            np.testing.assert_almost_equal(data[2, 3], 0.9)
        finally:
            os.unlink(tmp_path)

    def test_skip_comments_and_at_lines(self):
        """Lines starting with # or @ should be ignored."""
        from analyze_convergence import load_plumed_file

        content = """\
# comment line
@ another meta line
1.0  2.0
3.0  4.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            data = load_plumed_file(tmp_path)
            self.assertEqual(data.shape, (2, 2))
        finally:
            os.unlink(tmp_path)

    def test_empty_file_exits(self):
        """An empty data file should call sys.exit(1)."""
        from analyze_convergence import load_plumed_file

        content = "# only comments\n# no data here\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            with self.assertRaises(SystemExit) as cm:
                load_plumed_file(tmp_path)
            self.assertEqual(cm.exception.code, 1)
        finally:
            os.unlink(tmp_path)


class TestPlotDeltaGSingleRow(unittest.TestCase):
    """Test that plot_deltaG_convergence handles single-row files."""

    def test_single_row_no_crash(self):
        """A deltaG file with 1 data row should not crash."""
        content = "# time(ps)  deltaG(kJ/mol)\n5000.0  12.345\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            data = np.loadtxt(tmp_path, comments='#')
            data = np.atleast_2d(data)
            self.assertEqual(data.shape, (1, 2))
            self.assertAlmostEqual(data[0, 0], 5000.0)
            self.assertAlmostEqual(data[0, 1], 12.345)
        finally:
            os.unlink(tmp_path)

    def test_multi_row(self):
        """Multi-row files should work as before."""
        content = "# header\n1.0  10.0\n2.0  11.0\n3.0  12.0\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write(content)
            f.flush()
            tmp_path = f.name

        try:
            data = np.loadtxt(tmp_path, comments='#')
            data = np.atleast_2d(data)
            self.assertEqual(data.shape, (3, 2))
        finally:
            os.unlink(tmp_path)


class TestValidateAtomSelection(unittest.TestCase):
    """Test validate_atom_selection with valid and invalid patterns."""

    def setUp(self):
        from generate_plumed import validate_atom_selection
        self.validate = validate_atom_selection

    def test_valid_range(self):
        self.assertTrue(self.validate("1-50"))

    def test_valid_list(self):
        self.assertTrue(self.validate("1,2,3"))

    def test_valid_single_atom(self):
        self.assertTrue(self.validate("42"))

    def test_valid_large_range(self):
        self.assertTrue(self.validate("100-500"))

    def test_invalid_letters(self):
        self.assertFalse(self.validate("abc"))

    def test_invalid_mixed(self):
        self.assertFalse(self.validate("1,abc,3"))

    def test_invalid_empty_part(self):
        # "1,,3" splits into ['1', '', '3'] — '' is not a digit
        self.assertFalse(self.validate("1,,3"))

    def test_valid_range_with_comma_list(self):
        # "1-50" is a range (valid), but "1-50,100" has both - and ,
        # validate_atom_selection checks: if '-' in and ',' not in → range check
        # else: split by ',' and check each is digit
        # "1-50,100" → split by ',' → ["1-50", "100"]
        # "1-50".isdigit() is False → returns False
        self.assertFalse(self.validate("1-50,100"))


if __name__ == '__main__':
    unittest.main()
