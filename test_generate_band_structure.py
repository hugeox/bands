import generate_band_structure
import hf
import numpy as np
import unittest

class testOfCalculator(unittest.TestCase):
    def test_build_lattice(self):
        lattice = generate_band_structure.build_lattice(1.5)
        self.assertCountEqual(lattice,[[0,0,0],[1,0,1],[0,1,1],[-1,-1,1]])
    def test_neighbors(self):
        lattice = [[-1, -1, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1]]
        self.assertCountEqual(
                generate_band_structure.build_neighbor_table(
                    lattice),[[2,3,0],[2,1,1],[2,0,2]])
class testOfHartreeFock(unittest.TestCase):
    def test_shift_matrix(self):
        M = hf.shift_matrix(G=[-1,1],lattice = [[-1, -1, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1]])
        shifted = np.dot(M,[1,2,3,4,5,6,7,8])
        np.testing.assert_array_equal(shifted,[0,0,7,8,0,0,0,0])

if __name__ == '__main__':
    unittest.main()
