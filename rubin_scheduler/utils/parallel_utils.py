__all__ = ("SharedNumpyArray",)

from multiprocessing import shared_memory

import numpy as np


class SharedNumpyArray:
    """Class to hold a numpy array that can be shared with
    multiple processes without copying

    Origianlly from:
    https://e-dorigatti.github.io/python/2020/06/19/multiprocessing-large-objects.html

    Parameters
    ----------
    array : `np.array`
        Array to copy into shared memory.
    """

    def __init__(self, array):
        # Create the shared memory location of the same size of the array
        self._shared = shared_memory.SharedMemory(create=True, size=array.nbytes)
        # Save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = array.dtype, array.shape
        # Create a new numpy array that uses the shared memory we created.
        res = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shared.buf)
        # Copy data from the array to the shared memory. Numpy will
        # take care of copying everything in the correct format
        np.copyto(res, array)

    def read(self):
        """Read array without copy."""
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def copy(self):
        """Copy array"""
        return np.copy(self.read())

    def unlink(self):
        """Unlink when done with data"""
        self._shared.close()
        self._shared.unlink()
