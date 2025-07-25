import unittest

from rubin_scheduler.scheduler.surveys import ToOScriptedSurvey


class TestToO(unittest.TestCase):

    def test_too(self):
        survey = ToOScriptedSurvey([])
        ra1, dec1 = survey._tesselate([100, 101])
        # This should spin things
        ra2, dec2 = survey._tesselate([100, 101])

        assert ra2[0] != ra1[0]

        # If we ask for a single HEALpix
        # Should only give a pointing at
        # one location

        ra1, dec1 = survey._tesselate([100])

        assert len(ra1) == 1

        # Should be no change if we call again
        ra2, dec2 = survey._tesselate([100])

        assert ra1 == ra2


if __name__ == "__main__":
    unittest.main()
