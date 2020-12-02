import generate_band_structure
import unittest

class testOfCalculator(unittest.TestCase):
    def test_build_lattice():
        lattice = generate_band_structure.build_lattice(34)
        self.assertItemsEqual(lattice,[[0,0,0],[1,0,1],[0,1,1],[-1,-1,1]])

