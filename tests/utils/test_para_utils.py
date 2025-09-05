import unittest

import numpy as np

from rubin_scheduler.utils import SharedNumpyArray


class ParallelTools(unittest.TestCase):
    def test_shared_array(self):
        input_array = np.arange(100).reshape(10, 10)

        sharred_array = SharedNumpyArray(input_array)

        assert np.array_equal(input_array, sharred_array.read())
        assert np.array_equal(input_array, sharred_array.copy())

        sharred_array.unlink()


if __name__ == "__main__":
    unittest.main()
