__all__ = ["rotation_converter", "RotationConverter", "RotationConverterAuxtel"]

import numpy as np


def rotation_converter(telescope="rubin"):
    """Return the correct RotationConverter object."""
    if telescope.lower() == "rubin":
        return RotationConverter()
    elif telescope.lower() == "auxtel":
        return RotationConverterAuxtel()
    else:
        raise ValueError("Unknown telescope name")


def _wrap_180(in_angle):
    """Convert angle(s) to run from -180 to 180"""
    if np.size(in_angle) == 1:
        if in_angle > np.pi:
            result = in_angle - 2.0 * np.pi
            return result
        else:
            return in_angle
    else:
        indx = np.where(in_angle > np.pi)[0]
        result = in_angle + 0
        result[indx] = result[indx] - 2.0 * np.pi
        return result


class RotationConverter(object):
    """Class to convert between rotTelPos and rotSkyPos"""

    def rottelpos2rotskypos(self, rottelpos_in, pa):
        return np.degrees(self._rottelpos2rotskypos(np.radians(rottelpos_in), np.radians(pa)))

    def rotskypos2rottelpos(self, rotskypos_in, pa):
        result = self._rotskypos2rottelpos(np.radians(rotskypos_in), np.radians(pa))
        return np.degrees(result)

    def _rottelpos2rotskypos(self, rottelpos_in, pa):
        result = (pa - rottelpos_in - np.pi / 2) % (2.0 * np.pi)
        return result

    def _rotskypos2rottelpos(self, rotskypos_in, pa):
        result = (pa - rotskypos_in - np.pi / 2) % (2.0 * np.pi)
        # Enforce rotTelPos between -pi and pi
        return _wrap_180(result)


class RotationConverterAuxtel(RotationConverter):
    """Use a different relation for rotation angles on AuxTel"""

    def _rottelpos2rotskypos(self, rottelpos_in, pa):
        return (rottelpos_in - pa) % (2.0 * np.pi)

    def _rotskypos2rottelpos(self, rotskypos_in, pa):
        result = (rotskypos_in + pa) % (2.0 * np.pi)
        # Enforce rotTelPos between -pi and pi
        return _wrap_180(result)
