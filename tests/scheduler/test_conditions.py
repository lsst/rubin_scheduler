import unittest
import warnings
from inspect import signature

import numpy as np

from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.utils import SURVEY_START_MJD


class TestConditions(unittest.TestCase):
    def test_conditions(self):
        # Generate a Conditions object with things filled out
        mo = ModelObservatory()
        mo_conditions = mo.return_conditions()

        # Fresh empty conditions
        conditions = Conditions()

        # Find the arguments that need to be set
        auxtel_args = signature(conditions.set_auxtel_info)

        to_pass = []
        for key in auxtel_args.parameters.keys():
            if key != "kwargs":
                to_pass.append(getattr(mo_conditions, key))

        conditions.set_auxtel_info(*to_pass)

        for key in auxtel_args.parameters.keys():
            if key != "kwargs":
                if np.size(getattr(conditions, key)) == 1:
                    assert getattr(conditions, key) == getattr(mo_conditions, key)
                else:
                    assert np.array_equal(
                        getattr(conditions, key), getattr(mo_conditions, key), equal_nan=True
                    )

        # Again for the maintel attrs
        conditions = Conditions()
        maintel_args = signature(conditions.set_maintel_info)

        to_pass = []
        for key in maintel_args.parameters.keys():
            if key != "kwargs":
                to_pass.append(getattr(mo_conditions, key))

        conditions.set_maintel_info(*to_pass)

        for key in maintel_args.parameters.keys():
            if key != "kwargs":
                if (np.size(getattr(conditions, key)) == 1) | (isinstance(getattr(conditions, key), list)):
                    assert getattr(conditions, key) == getattr(mo_conditions, key)
                else:
                    assert np.array_equal(
                        getattr(conditions, key), getattr(mo_conditions, key), equal_nan=True
                    )

        # check we can set some arbitrary attributes
        conditions = Conditions()
        conditions.set_attrs(mjd=62511, moon_alt=0.1)

        with self.assertWarns(Warning):
            conditions.set_attrs(not_an_attribute=10)

        # Test that conditions.night is set as expected
        current_mjd = 60914.73382326572
        conditions.mjd = current_mjd
        print(conditions.mjd, conditions.night)
        expected_night = np.floor(current_mjd - 0.5) - np.floor(SURVEY_START_MJD - 0.5)
        self.assertEqual(expected_night, conditions.night)
        # Test that warning is raised if you try to set night directly
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            conditions.night = 10
            # Verify deprecation warning
            assert len(w) >= 1
            assert issubclass(w[-1].category, (DeprecationWarning, FutureWarning))


if __name__ == "__main__":
    unittest.main()
