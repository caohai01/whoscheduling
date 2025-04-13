from astroplan import Constraint, min_best_rescale, AltitudeConstraint, max_best_rescale, AtNightConstraint, \
    AirmassConstraint, MoonSeparationConstraint, Observer, moon_illumination
from astroplan.constraints import _get_moon_data, _make_cache_key, _get_altaz
from astropy import units as u
import numpy as np
from astropy.coordinates import EarthLocation, get_sun, get_body, SkyCoord


class MagnitudeConstraint(Constraint):
    """
    Constrain the magnitude of the target.

    Parameters
    ----------
    min : float or `None`
        Minimum magnitude of the target (inclusive). `None` indicates no limit.
    max : float or `None`
        Maximum magnitude of the target (inclusive). `None` indicates no limit.
    boolean_constraint : bool
        If True, the constraint is treated as a boolean (True for within the
        limits and False for outside).  If False, the constraint returns a
        float on [0, 1], where 1 is the min magnitude and 0 is the max magnitude.
    """

    def __init__(self, min_val=None, max_val=None, boolean_constraint=True):
        if min_val is None:
            self.min = 13
        else:
            self.min = min_val
        if max_val is None:
            self.max = 23
        else:
            self.max = max_val

        self.boolean_constraint = boolean_constraint

    def compute_constraint(self, times, observer, targets):

        mag = np.array([t.mag if hasattr(t, 'mag') else self.min for t in targets])
        if self.boolean_constraint:
            lowermask = self.min <= mag
            uppermask = mag <= self.max
            return lowermask & uppermask
        else:
            return min_best_rescale(self._compute(times, observer, targets), self.min, self.max)

    def _compute(self, times, observer, targets):
        targets_altaz = _get_altaz(times, observer, targets)
        targets_airmass = targets_altaz['altaz'].secz.value
        targets_airmass[targets_airmass < 0] = 0.0
        targets_airmass[targets_airmass > 4] = 4.0
        k = 0.24  # extinction coefficient in Xinglong, weihai 0.24
        mag = [t.mag if hasattr(t, 'mag') else self.min for t in targets]
        apparent_mag = mag + k * targets_airmass

        return apparent_mag


class SkyBrightnessConstraint(Constraint):
    """
       Constrain the sky brightness of the target, moon dominate.

       Parameters
       ----------
       min : float or `None`
           Minimum magnitude of the sky (inclusive). `None` indicates no limit.
       max : float or `None`
           Maximum magnitude of the sky (inclusive). `None` indicates no limit.
       boolean_constraint : bool
           If True, the constraint is treated as a boolean (True for within the
           limits and False for outside).  If False, the constraint returns a
           float on [0, 1], where 1 is the max and 0 is the min.
       """

    def __init__(self, min_val=None, max_val=None, boolean_constraint=False, ephemeris=None):
        if min_val is None:
            self.min = 0.0
        else:
            self.min = min_val
        if max_val is None:
            self.max = 1
        else:
            self.max = max_val

        self.boolean_constraint = boolean_constraint
        self.ephemeris = ephemeris
        self._moon_cache = {}  # cache moon

    def compute_constraint(self, times, observer, targets):
        sky_bright = self._get_moon_brightness_xl(times, observer, targets)
        if self.boolean_constraint:
            lowermask = self.min <= sky_bright
            uppermask = sky_bright <= self.max
            return lowermask & uppermask
        else:
            return sky_bright

    def nL_to_mag(self, nL):
        # Conversion factor from nanoLambert to the magnitude scale zero point
        # Compute magnitude per square arcsecond
        mag_per_arcsec2 = 26.33 - 2.5 * np.log10(nL)

        return mag_per_arcsec2

    def mag_to_nL(self, mag_per_arcsec2):
        conversion_factor = 34.08
        #nL = conversion_factor * np.exp(20.7233 - 0.92104 * mag)
        nL = 10 ** ((26.33 - mag_per_arcsec2) / 2.5)
        return nL

    def _get_moon_brightness_xl(self, times, observer, targets):
        """
        implementation of Moon night sky brightness simulation for the Xinglong station
        :param times:
        :param observer:
        :param targets:
        :return:
        """
        key = _make_cache_key(times, targets)
        if key in self._moon_cache:
            return self._moon_cache[key]
        moon = get_body('moon', times, location=observer.location, ephemeris=self.ephemeris)
        sun = get_sun(times)
        moon_altaz = _get_altaz(times, observer, moon)
        targets_altaz = _get_altaz(times, observer, targets)
        elongation = sun.separation(moon, origin_mismatch="ignore")
        moon_phase_angle = np.arctan2(sun.distance * np.sin(elongation),
                                      moon.distance - sun.distance * np.cos(elongation))
        t = (1 + np.cos(moon_phase_angle)) / 2.0
        moon_illu = t.value  # Fraction of moon illuminated[0, 1]

        moon_airmass = moon_altaz['altaz'].secz.value
        # 当z高于75°时，直接取值4
        moon_airmass[moon_airmass < 1.0] = 1.0
        moon_airmass[moon_airmass > 4] = 4.0

        targets_airmass = targets_altaz['altaz'].secz.value
        targets_airmass[targets_airmass < 1.0] = 1.0
        targets_airmass[targets_airmass > 4] = 4.0

        # moon phase angle a in degrees
        a = moon_phase_angle.to(u.degree).value

        sep = moon.separation(targets, origin_mismatch="ignore")  # degrees,
        # p is the scattering angle defined as the angular separation between the moon and the target
        p = sep.value # angular separation between the moon and the target, in degrees
        p[p > 90] = 90
        p[p < 15] = 0.0

        k = 0.24  # extinction coefficient in Xinglong, weihai 0.24
        airmass = 1.414  # the airmass
        # the airmass for the moon and the target
        # I is the brightness of the moon outside the atmosphere
        # mag_v = 16.57 + -12.73 + 1.49 * moon_phase_angle + 0.043 * (moon_phase_angle**4)
        I = np.power(10, -0.4*(3.84 + 0.026 * a + 4e-9 * (a**4)))
        # f(p) is the scattering function at scattering angle p
        # p is the scattering angle defined as the angular separation between the moon and the target
        # p = 15 # angular separation between the moon and the target, in degrees
        def f(p):
            # f_r and f_m are the Rayleigh and Mie scattering model
            pr = np.radians(p)
            cosp = np.cos(pr)
            f_r = np.power(10, 5.36) * (1.06 + cosp * cosp)
            f_m = np.power(10, (6.15-p/40.0))

            # else:
            #     f_m = 6.2e7 / (p * p)

            # PA is a scale factor for Mie scattering and PB is a scale factor for Rayleigh scattering
            # fitted value 1.5 and 0.9 see Fig. 5
            PA = 1.5
            PB = 0.9
            return PA * f_r + PB * f_m

        # B_z is the dark time sky brightness at zenith in nanoLamberts (nL)
        # which can be converted to magnitude V with equation (27) in Garstang (1989)
        # V= 21.4 mag/arcsec^2.
        B_z = 93.76
        B_0 = B_z * np.power(10, -0.4 * k * (targets_airmass - 1)) * targets_airmass
        B_moon = f(p) * I * np.power(10, -0.4*k*moon_airmass) * (1 - np.power(10, -0.4*k*targets_airmass))
        # nanoLamberts to magnitude -> -2.5 * math.log10(nanolambert / 3.142)
        return min_best_rescale(B_0 + B_moon, 90.0, 3000)

    def _get_moon_brightness(self, times, observer, targets):
        # key = hash(times) + hash(targets) + len(times) + len(targets)
        key = _make_cache_key(times, targets)
        if key in self._moon_cache:
            return self._moon_cache[key]
        moon = get_body('moon', times, location=observer.location, ephemeris=self.ephemeris)
        sun = get_sun(times)
        altaz = observer.altaz(times, moon, grid_times_targets=False)
        elongation = sun.separation(moon, origin_mismatch="ignore")
        moon_phase_angle = np.arctan2(sun.distance * np.sin(elongation),
                                      moon.distance - sun.distance * np.cos(elongation))
        dark_part = (1 + np.cos(moon_phase_angle)) / 2.0
        moon_illu = 1.0 - dark_part.value  # Fraction of moon illuminated[0, 1]
        moon_alt = altaz.alt.value
        # illu = min_best_rescale(illumination, 0.0, 1.0, 0.0)
        # origin_mismatch="ignore" to disableNonRotationTransformationWarning: transforming other coordinates from

        sep = moon.separation(targets, origin_mismatch="ignore")  # degrees,
        moon_sep = sep.value
        cosp = np.cos(np.radians(moon_sep))
        f_r = np.power(10, 5.36) * (1.06 + cosp * cosp)
        f_m = np.power(10, (6.15 - moon_sep / 40.0))
        f_s = f_r + f_m
        sep_scaled = max_best_rescale((moon_sep - 20.0 * moon_illu), 10.0, 90.0, 1.0)
        mb = 1.0 / (1.0 + np.exp(moon_illu / sep_scaled)) - moon_alt / 900.0
        #mb = max_best_rescale(sep_scaled * (1.0 - moon_illu), self.min, self.max)
        mb = max_best_rescale(mb, self.min, self.max)
        self._moon_cache.update({key: mb})
        return mb


class ExtinctionConstraint(Constraint):
    """
    Constrain the Atmospheric Extinction.

    Parameters
    ----------
    min : float or `None`
        Minimum magnitude of the target (inclusive). `None` indicates no limit.
    max : float or `None`
        Maximum magnitude of the target (inclusive). `None` indicates no limit.
    boolean_constraint : bool
        If True, the constraint is treated as a boolean (True for within the
        limits and False for outside).  If False, the constraint returns a
        float on [0, 1], where 1 is the min magnitude and 0 is the max magnitude.
    """

    def __init__(self, val,  min_val=None, max_val=None):
        if min_val is None:
            self.min = 0.0
        else:
            self.min = min_val
        if max_val is None:
            self.max = 1.0
        else:
            self.max = max_val
        self.value = val
        self.boolean_constraint = False

    def compute_constraint(self, times, observer, targets):

        return min_best_rescale(np.array([self.value]), self.min, self.max)


class CombinedConstraints(Constraint):
    """
    计算多个约束点积
    """

    def __init__(self, constraints=None):
        if constraints is None:
            self.constraints = [
                AtNightConstraint.twilight_civil(),
                AltitudeConstraint(min=15 * u.degree),
            ]
        else:
            self.constraints = constraints

    @classmethod
    def SkyBrightnessAirmassAltitude(cls):
        """
        Consider nighttime as time between civil twilights (-6 degrees).
        """
        constraints = [
            #AtNightConstraint.twilight_civil(),
            #AltitudeConstraint(min=15 * u.degree),
            #AirmassConstraint(max=4, boolean_constraint=False),
            SkyBrightnessConstraint(),
            # MagnitudeConstraint(boolean_constraint=False),
        ]
        return [cls(constraints=constraints)]

    def _compute(self, times, observer, targets):

        if not targets.isscalar:  # only one target per time
            targets = targets[0]

        constraint_score = np.ones((1, len(times)))
        add_constraints = []
        for constraint in self.constraints:
            if type(constraint) is not SkyBrightnessConstraint:
                constraint_score *= constraint(observer, targets, times, grid_times_targets=True)
            else:
                add_constraints.append(constraint) # process later

        for constraint in add_constraints:
            constraint_score *= constraint(observer, targets, times, grid_times_targets=True)

        return constraint_score

    def compute_constraint(self, times, observer, targets):
        # 实现分数自定义组合，不仅仅是相乘
        return self._compute(times, observer, targets)
