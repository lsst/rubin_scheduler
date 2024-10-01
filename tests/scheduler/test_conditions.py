import unittest
from inspect import signature

from rubin_scheduler.scheduler.features import Conditions
from rubin_scheduler.scheduler.model_observatory import ModelObservatory


class TestConditions(unittest.TestCase):

    def test_conditions(self):
        # Generate a Conditions object with things filled out
        mo = ModelObservatory()
        mo_conditions = mo.return_conditions()

        conditions = Conditions()

        # Find the arguments that need to be set
        auxtel_args = signature(conditions.set_auxtel_info)
        
        to_pass = {}
        for key in auxtel_args.parameters.keys():
            to_pass[key] = getattr(mo_conditions, key)

        conditions.set_auxtel_info(**to_pass)


if __name__ == "__main__":
    unittest.main()
