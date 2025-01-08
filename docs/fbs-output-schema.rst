.. py:currentmodule:: rubin_scheduler

.. _fbs-output-schema:

=======================
Scheduler Output Schema
=======================

When running in simulation mode, the scheduler outputs a sqlite database
containing the pointing history of the telescope, along with information
about the conditions of each observation (visit).
The table below describes the columns in the `observations` table.

All values are for the center of the field of view (e.g., airmass, altitude, etc)

.. list-table:: Opsim Schema
   :widths: 40 25 100
   :header-rows: 1

   * - Column
     - Units
     - Description
   * - observationId
     - integer
     - Simple counter for the observation.
   * - fieldRA
     - degrees
     - The Right Ascension of the boresight.
   * - fieldDec
     - degrees
     - The Declination of the boresight.
   * - observationStartMJD
     - MJD
     - The MJD of the start of the observation (UTC).
   * - flush_by_mjd
     - MJD
     - The time by which to drop the queued observation from consideration.
   * - visitExposureTime
     - seconds
     - Total on-sky exposure time.
   * - band
     - string
     - The name of the bandpass in use for the observation, typically one of ugrizy.
   * - filter
     - string
     - The name of the physical filter in use for the observation.
   * - rotSkyPos
     - degrees
     - The orientation of the sky in the focal plane measured as the angle between ICRF North on the sky and the "up" direction in the focal plane. Corresponds to 'skyAngle'.
   * - rotSkyPos_desired
     - degrees
     - The rotSkyPos requested by the observation. Most likely to match rotSkyPos, but if the angle is unreachable, then backup values will be used.
   * - numExposures
     - integer
     - The number of exposures in the visit.
   * - airmass
     - unitless
     - The airmass of the visit, at the center of the field at the start of the visit.
   * - seeingFwhm500
     - arcseconds
     - The seeing FWHM at zenith at 500 nm. The atmospheric contribution to the IQ.
   * - seeingFwhmEff
     - arcseconds
     - The FWHM of the single-gaussian PSF that results in equivalent SNR for a point source, compared to the full realized PSF.
   * - seeingFwhmGeom
     - arcseconds
     - The FWHM of a measured PSF, a measurement of the physical size of the (non-gaussian) PSF.
   * - skyBrightness
     - mag/arcsecond^2
     - The skybrightness in the filter in use at the center of the exposure. Predicted from the skybrightness_pre module.
   * - night
     - integer
     - The night of the survey for the observation.
   * - slewTime
     - seconds
     - The time required to slew to this position on the sky. Includes filter change and readout, as well as slew and settle time.
   * - visitTime
     - seconds
     - The total wall-clock time required for a visit. Longer than visitExposureTime if multiple exposures within a visit were acquired.
   * - slewDistance
     - degrees
     - Distance required to slew to this pointing from the previous pointing.
   * - fiveSigmaDepth
     - magnitudes
     - The magnitude of a five-sigma point source detection in the visit.
   * - altitude
     - degrees
     - Altitude of the field at the start of the observation.
   * - azimuth
     - degrees
     - Azimuth of the field at the start of the observation.
   * - paraAngle
     - degrees
     - Parallactic angle of the observation, angle between True North and zenith.
   * - pseudoParaAngle
     - degrees
     - Angle between ICRF North and zenith. See https://smtn-019.lsst.io/v/DM-44258/index.html
   * - cloud
     - float
     - Fraction of sky which is cloudy. Simply used as an indicator to shut the telescope.
   * - moonAlt
     - degrees
     - Altitude of the Moon.
   * - sunAlt
     - degrees
     - Altitude of the Sun.
   * - scheduler_note
     - string
     - Descriptive comment about how the observations were scheduled, for use by the FBS.
   * - target_name
     - string
     - Descriptive name for the target. Only used for DDFs, ToOs, or other special targets. Should translate to target_name in the headers/ConsDB.
   * - observationStartLST
     - degrees
     - Local Sidereal Time at the start of the observation.
   * - rotTelPos
     - degrees
     - Angle between zenith and "up" in the camera. Camera rotator angle.
   * - rotTelPos_backup
     - degrees
     - Angle specifying a backup for rotSkyPos_desired and rotTelPos.
   * - moonAz
     - degrees
     - Azimuth of the Moon.
   * - sunAz
     - degrees
     - Azimuth of the Sun.
   * - sunRA
     - degrees
     - Right Ascension of the Sun.
   * - sunDec
     - degrees
     - Declination of the Sun.
   * - moonRA
     - degrees
     - Right Ascension of the Moon.
   * - moonDec
     - degrees
     - Declination of the Moon.
   * - moonDistance
     - degrees
     - Distance between the field pointing and the Moon.
   * - solarElong
     - degrees
     - Solar Elongation of the field.
   * - moonPhase
     - float
     - Lunar illumination, ranging from 0 (new moon) to 100 (full moon).
   * - cummTelAz
     - degrees
     - Cumulative azimuth of the telescope mount, tracks cable wrap.
   * - target_id
     - integer
     - Integer added by the `CoreScheduler`.
   * - observation_reason
     - string
     - The reason for the observation. Identifier for DM. Translates to observation_reason in the headers/consdb.
   * - science_program
     - string
     - The science program for the observation. This will typically track to a JSON BLOCK to execute for the scheduler and translates to science_program in the headers/consdb. An identifier for Scheduler and DM.
