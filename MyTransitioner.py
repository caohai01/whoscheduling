from astroplan import TransitionBlock
from astropy import units as u
from math import sqrt

from astropy.coordinates import AltAz
from astropy.units import Quantity


class MyTransitionBlock(TransitionBlock):
    def __init__(self, components, start_time=None):
        super().__init__(components, start_time)

        self.duration = 0 * u.second
        self._components = None
        self.components = components


    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, val):
        duration = 0 * u.second
        self._components = {}

        for k, v in val.items():
            if k.startswith('_'):
                continue

            self._components.update({k: v})

            if not isinstance(v, Quantity):
                continue

            if v > duration:
                duration = v

        self.duration = duration


class MyTransitioner(object):
    """
    A class that defines how to compute transition times from one block to
    another.
    """
    u.quantity_input(telescope_slew_rate=u.deg / u.second)
    u.quantity_input(dome_slew_rate=u.deg / u.second)

    def __init__(self, telescope_slew_rate=None, dome_slew_rate=None, filter_switch_rate=None, filters_count=8):

        self.telescope_slew_rate = telescope_slew_rate
        self.telescope_slew_acc_rate = 5.0 * u.deg / u.second / u.second  # deg / sec^2
        self.dome_slew_rate = dome_slew_rate
        self.filter_switch_rate = filter_switch_rate
        self.filters_count = filters_count

    def _get_telescope_slew_time(self, from_coord, to_coord):
        # 赤经
        RA1, Dec1 = from_coord.ra, from_coord.dec
        RA2, Dec2 = to_coord.ra, to_coord.dec
        RA_Rate_Max = self.telescope_slew_rate
        Dec_Rate_Max = self.telescope_slew_rate
        RA_Acc_Rate = Dec_Acc_Rate = self.telescope_slew_acc_rate

        dRA = abs(RA2 - RA1)
        if dRA > 180.0 * u.deg:
            dRA = 360 * u.deg - dRA
        if dRA > RA_Rate_Max * RA_Rate_Max / RA_Acc_Rate:  # 距离够远，能够达到最大速度 RA_Rate_Max
            tRA = dRA / RA_Rate_Max + RA_Rate_Max / RA_Acc_Rate
        else:
            tRA = 2 * sqrt((dRA / RA_Acc_Rate).value) * u.second  # sqrt cannot work for s^2

        # 赤纬
        dDec = abs(Dec2 - Dec1)
        if dDec > Dec_Rate_Max * Dec_Rate_Max / Dec_Acc_Rate:
            tDec = dDec / Dec_Rate_Max + Dec_Rate_Max / Dec_Acc_Rate
        else:
            tDec = 2 * sqrt((dDec / Dec_Acc_Rate).value) * u.second  # sqrt cannot work for s^2

        return max(tRA, tDec)

    def _get_dome_slew_time(self, from_az, to_az):
        if self.dome_slew_rate.value <= 0: # none dome
            return 0 * u.second

        sep = abs(to_az - from_az)
        if sep > 180.0 * u.deg:  # revers direction
            sep = 360.0 * u.deg - sep
        return sep / self.dome_slew_rate

    def _get_filter_switch_time(self, form_filter_pos, to_filter_pos):
        if self.filter_switch_rate <= 0:
            return 0 * u.second
        sep = abs(form_filter_pos - to_filter_pos)
        if sep > self.filters_count // 2:
            sep -= self.filters_count // 2
        return (sep / self.filter_switch_rate) * u.second

    def __call__(self, from_block, to_block, start_time, observer):
        """
        Determines the amount of time needed to transition from one observing
        block to another.  This uses the parameters defined in
        ``self.instrument_reconfig_times``.

        Parameters
        ----------
        from_block : `~astroplan.scheduling.ObservingBlock` or None
            The initial configuration/target
        to_block : `~astroplan.scheduling.ObservingBlock` or None
            The new configuration/target to transition to
        start_time : `~astropy.time.Time`
            The time the transition should start
        observer : `astroplan.Observer`
            The observer at the time

        Returns
        -------
        transition : `~astroplan.scheduling.TransitionBlock` or None
            A transition to get from ``from_block`` to ``to_block`` or `None` if
            no transition is necessary
        """
        components = {}
        if from_block is not None and to_block is not None:
            if from_block.target != to_block.target:
                from_coord = from_block.target.coord
                to_coord = to_block.target.coord
                if self.telescope_slew_rate is not None:
                    telescope_slew_time = self._get_telescope_slew_time(from_coord, to_coord)
                    components['telescope_slew'] = telescope_slew_time
                if self.dome_slew_rate is not None:
                    altaz_frame = AltAz(location=observer.location, obstime=start_time)
                    from_altaz = from_coord.transform_to(altaz_frame)
                    to_altaz = to_coord.transform_to(altaz_frame)
                    dome_slew_time = self._get_dome_slew_time(from_altaz.az, to_altaz.az)
                    components['dome_slew'] = dome_slew_time

            from_conf = from_block.configuration
            to_conf = to_block.configuration
            key = 'filters_position'
            if key in from_conf and key in to_conf:
                last_pos = from_conf[key][-1]
                to_pos = to_conf[key][0]
                if last_pos != to_pos:
                    filter_switch_time = self._get_filter_switch_time(last_pos, to_pos)
                    key2 = 'filters'
                    components[f'filter {from_conf[key2]} -> {to_conf[key2]}'] = filter_switch_time

        if components:
            return MyTransitionBlock(components, start_time)
        else:
            return None
