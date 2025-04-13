import numpy as np
from astroplan import Observer
from astroplan.constraints import _get_altaz
from astropy.coordinates import get_body, get_sun, SkyCoord, EarthLocation
from astropy.time import Time
from astropy import units as u


def get_exposure_time(snr, mag):
    diam = 15  # cm
    std_flux = 7220
    eta = 0.4
    seeing = 0.6
    delta = 2.3  # read noise e-
    A = np.pi * diam * diam / 4
    k = 0.185
    X = 1.414
    m = mag + k * X
    F_target = std_flux * np.power(10, -0.4 * m) * A * eta
    avg_sky_mag = 19
    ms = avg_sky_mag + k * X
    F_sky = std_flux * np.power(10, -0.4 * ms) * A * eta * seeing ** 2

    a = F_target ** 2
    b = -snr ** 2 * (F_target + F_sky)
    c = -snr ** 2 * delta ** 2

    t = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2 * a
    return t

def get_exposure_time_I(snr, mag, coord, time):
    """
    按照天顶角45度，大气消光0.2系数，天光背景恒定
    :param snr:
    :param mag:
    :param band:
    :return:
    """
    ccd_pixel_size = 13.5  # um
    ccd_gain = 1.8
    ccd_bin = 1
    ccd_noise = 15.0
    ccd_dark = 0.02

    tel_diam = 100  # 直径 cm
    tel_f = 800  # 焦距 cm
    tel_aperture = tel_diam / tel_f * 100  # 孔径，焦比倒数
    tel_efficiency = 0.6  # 光学效率
    tel_seeing = 1.75  # 当地视宁度

    zero_point = 3780  # 仪器零点1 https://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/pet/magtojy/
    # U B V R

    ext_coefficient = 0.185
    std_flux = 67542
    airmass = 1.4  # 大气质量，高度在45度左右，计算出1.4
    avg_sky_brightness = 18.5

    m = mag + ext_coefficient * airmass
    location = EarthLocation.from_geodetic(122.04961 * u.deg, 37.53592 * u.deg, 100 * u.m)
    B01, tX = _get_moon_brightness_xl(time, Observer(location=location), coord)
    avg_sky_brightness += (B01 * -12.73)
    ms = avg_sky_brightness
    # 1 sbu = 10^-9 erg s^-1 cm^-2 Å^-1 sr^-1
    # (ccd_pixel_size * 10^-4)^2 * 10^9
    # ms = ms * tel_efficiency * tel_aperture * tel_aperture * (ccd_pixel_size * ccd_pixel_size * 10) * np.pi / 4.0
    scale = 206265.0 / tel_f * (ccd_pixel_size / 10000)
    pixel_count = np.pi * (tel_seeing / scale) * (tel_seeing / scale)

    Ns = np.pi * tel_diam * tel_diam / 4 * std_flux * tel_efficiency * pow(10, -0.4 * m)
    Nb = np.pi * tel_diam * tel_diam / 4 * std_flux * tel_efficiency * pow(10, -0.4 * ms) * pixel_count
    Nd = ccd_dark * pixel_count

    a = Ns * Ns
    b = -snr * snr * (Ns + Nb + Nd)
    c = -snr * snr * pixel_count * ccd_noise * ccd_noise
    exp = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if exp < 0.01:  # 限制在0.01和360秒之间
        exp = 0.01
    if exp > 120 :
        exp = 120 # 无法观测
    return exp
def get_exposure_time_W(snr, mag, coord, time):
    """
    按照天顶角45度，大气消光0.2系数，天光背景恒定
    :param snr:
    :param mag:
    :param band:
    :return:
    """
    ccd_pixel_size = 9  # um
    ccd_gain = 1.8
    ccd_bin = 1
    ccd_noise = 2.3
    ccd_dark = 0.002

    tel_diam = 15  # 直径 cm
    tel_f = 21  # 焦距 cm f/1.4
    tel_aperture = tel_diam / tel_f * 100  # 孔径，焦比倒数
    tel_efficiency = 0.8  # 光学效率
    seeing = 0.6  # 当地视宁度， arcsec

    zero_point = 3780  # 仪器零点1 https://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/pet/magtojy/
    # U B V R

    ext_coefficient = 0.185
    std_flux = 10000
    airmass = 1.414  # 大气质量，高度在45度左右，计算出1.4
    avg_sky_brightness = 19

    m = mag + ext_coefficient * airmass
    ms = avg_sky_brightness + ext_coefficient * airmass
    # 1 sbu = 10^-9 erg s^-1 cm^-2 Å^-1 sr^-1
    # (ccd_pixel_size * 10^-4)^2 * 10^9
    # ms = ms * tel_efficiency * tel_aperture * tel_aperture * (ccd_pixel_size * ccd_pixel_size * 10) * np.pi / 4.0
    scale = ccd_pixel_size / tel_f / 10 * 206.265  # arcsec/pixel
    pixel_count = np.pi * (seeing / scale) * (seeing / scale)
    area = np.pi * tel_diam * tel_diam / 4
    Ns = area * std_flux * tel_efficiency * pow(10, -0.4 * m)
    Nb = area * std_flux * tel_efficiency * pow(10, -0.4 * ms) * pixel_count
    Nd = ccd_dark * pixel_count

    a = Ns * Ns
    b = -snr * snr * (Ns + Nb + Nd)
    c = -snr * snr * pixel_count * ccd_noise * ccd_noise
    exp = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    # if exp < 0.1:  # 限制在0.01和120秒之间
    #     exp = 0.1
    # if exp > 120 - 1:
    #     exp = 120 - 1
    return exp


def get_moon_scatter(target, time):
    moon = get_body('moon', time)
    sun = get_sun(time)
    #moon_altaz = _get_altaz(time, observer, moon)
    #targets_altaz = _get_altaz(time, observer, target)
    elongation = sun.separation(moon, origin_mismatch="ignore")
    moon_phase_angle = np.arctan2(sun.distance * np.sin(elongation), moon.distance - sun.distance * np.cos(elongation))

    t = (1 + np.cos(moon_phase_angle)) / 2.0
    moon_illu = t.value  # Fraction of moon illuminated[0, 1]
    sep = moon.separation(target, origin_mismatch="ignore").to(u.rad)  # rad,
    if sep < 0.175 * u.rad:  # 10度
        return -12.73
    elif sep > 1.57 * u.rad:  # 90度
        return 0
    sep = 1.57 * u.rad - sep
    moon_mag = (-12.73 - 3.84) * moon_illu + 3.84
    scat = ((4 * sep ** 2) / 9.87) * moon_mag
    return scat.value


def _get_moon_brightness_xl(times, observer, targets):
    """
    implementation of Moon night sky brightness simulation for the Xinglong station
    :param times:
    :param observer:
    :param targets:
    :return:
    """
    moon = get_body('moon', times)
    sun = get_sun(times)
    moon_altaz = _get_altaz(times, observer, moon)
    targets_altaz = _get_altaz(times, observer, targets)
    elongation = sun.separation(moon, origin_mismatch="ignore")
    moon_phase_angle = np.arctan2(sun.distance * np.sin(elongation), moon.distance - sun.distance * np.cos(elongation))
    t = (1 + np.cos(moon_phase_angle)) / 2.0
    moon_illu = t.value  # Fraction of moon illuminated[0, 1]

    moon_airmass = moon_altaz['altaz'].secz.value
    # 当z高于75°时，直接取值4
    if moon_airmass > 4:
        moon_airmass = 4.0
    elif moon_airmass < 1:
        moon_airmass = 1.0

    targets_airmass = targets_altaz['altaz'].secz.value
    if targets_airmass > 4:
        targets_airmass = 4.0
    elif targets_airmass < 1:
        targets_airmass = 1.0

    # moon phase angle a in degrees
    a = moon_phase_angle.to(u.degree).value

    sep = moon.separation(targets, origin_mismatch="ignore")  # degrees,
    # p is the scattering angle defined as the angular separation between the moon and the target
    p = sep.value  # angular separation between the moon and the target, in degrees
    if p > 90:
        p = 90
    elif p < 15:
        p = 0.0

    k = 0.24  # extinction coefficient in Xinglong, weihai 0.24
    # airmass = 1.414  # the airmass  the airmass for the moon and the target
    # I is the brightness of the moon outside the atmosphere
    # mag_v = 16.57 + -12.73 + 1.49 * moon_phase_angle + 0.043 * (moon_phase_angle**4)
    I = np.power(10, -0.4 * (3.84 + 0.026 * a + 4e-9 * (a ** 4)))

    # f(p) is the scattering function at scattering angle p
    # p is the scattering angle defined as the angular separation between the moon and the target
    # p = 15 # angular separation between the moon and the target, in degrees
    def f(p):
        # f_r and f_m are the Rayleigh and Mie scattering model
        pr = np.radians(p)
        cosp = np.cos(pr)
        f_r = np.power(10, 5.36) * (1.06 + cosp ** 2)
        f_m = np.power(10, (6.15 - p / 40.0))

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
    B_moon = f(p) * I * np.power(10, -0.4 * k * moon_airmass) * (1 - np.power(10, -0.4 * k * targets_airmass))
    # nanoLamberts to magnitude -> -2.5 * math.log10(nanolambert / 3.142)
    B_total = B_0 + B_moon
    # normalize B_total(90, 3090) to 0-1
    if B_total < 90:
        B_total = 90
    elif B_total > 3090:
        B_total = 3090
    B_1 = (B_total - 90) / (3090 - 90)
    return B_1, targets_airmass


def get_exposure_time_xgd(snr, mag, target, time):
    diam = 15  # cm
    pixel_size = 9  # um / pixel
    focus_length = 21  # cm
    scale = pixel_size / (focus_length * 10000) * 206265  # arcsec / pixel，调节焦距为1.05cm，以匹配成像质量
    std_flux = 7.22 * 1e5  # photons / second / cm^2, 白光波长范围是380~780nm，此处按照570nm，
    std_flux /= 0.67  # 转换为G波段流量，G波段与白光之间的关系大约是0.3~0.7之间的倍数，取中间值
    eta = 0.4
    seeing = 1.2  # arcsec
    read_noise = 3.97  # e-/pixel
    k = 0.24  # mag / airmass
    X = 1.414  #  45 deg, sec(z)
    # m = mag + k * X
    A = np.pi * (diam / 2) ** 2  # cm^2
    SA = np.pi * (seeing / 2) ** 2  # arcsec^2
    pixel_count = SA / (scale * scale)

    base_sky_mag = 19.2
    location = EarthLocation.from_geodetic(74.893245 * u.deg, 38.335449 * u.deg, 4490 * u.m)
    B01, tX = _get_moon_brightness_xl(time, Observer(location=location), target)
    base_sky_mag += (B01 * -12.73)
    # mb = get_moon_scatter(target, time)
    # if mb <= -12.73:
    #     base_sky_mag = -12.73
    # else:
    #     base_sky_mag += mb
    X = tX
    m = mag + k * X
    F_target = std_flux * 10 ** (-0.4 * m) * eta * A
    mag_sky = base_sky_mag + k * X
    F_sky = std_flux * 10 ** (-0.4 * mag_sky) * SA * eta * A + read_noise ** 2 * pixel_count

    t = snr ** 2 * (1 + F_sky / F_target) / F_target
    if t < 0.1:
        t = 0.1
    elif t > 360:
        t = -1 # 无法观测
    return t


exposure_time_dicts = {
    'whow': get_exposure_time_I,
    'mstg': get_exposure_time_xgd,
}

if __name__ == '__main__':
    start_time = Time("2025-03-20 14:00:00")  # 起始时间UTC
    target = SkyCoord(ra=52.007227, dec=40.014738, unit='deg')
    snr = 100
    mag = 12.6
    # location = EarthLocation.from_geodetic(74.893245 * u.deg, 38.335449 * u.deg, 4490 * u.m)
    t1 = get_exposure_time_xgd(snr, mag, target, start_time)
    # _get_moon_brightness_xl(start_time, Observer(location=location), target)
    print(f'xgd:\nsnr:{snr}\nmag:{mag}\ntime:{t1:.3f}s\n')

    t2 = get_exposure_time_I(snr, mag, target, start_time)
    # _get_moon_brightness_xl(start_time, Observer(location=location), target)
    print(f'who\nsnr:{snr}\nmag:{mag}\ntime:{t2:.3f}s\n')
