import unittest

from rubin_scheduler.utils import (
    bright_filter_colors,
    get_filter_colors,
    get_filter_linestyles,
    get_filter_symbols,
)


class PlotUtilsTest(unittest.TestCase):

    def test_plot_utils(self):
        temp = get_filter_colors()
        assert isinstance(temp, dict)

        temp = get_filter_symbols()
        assert isinstance(temp, dict)

        temp = get_filter_linestyles()
        assert isinstance(temp, dict)

        temp = bright_filter_colors()
        assert isinstance(temp, dict)


if __name__ == "__main__":
    unittest.main()
