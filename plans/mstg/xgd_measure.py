import json
from astropy.io.fits import open as fits_open, writeto as fits_save, getheader
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, ApertureStats, CircularAnnulus, aperture_photometry
import numpy as np
from os import path, system, remove, mkdir
import glob
import argparse


# 过滤选择小光电的拖长文件，只保留观测正常的文件
# 按照观测策略，同一个目标中间观测的文件正常，其他排除
def select_files(source_path):
    zip_files = glob.glob(path.join(source_path, '*.fits.gz'))
    if len(zip_files) < 1:
        return []

    # 按照文件名第二个下划线后面的编号分组， 文件名格式：20241228140717725_7836_520448_DFLYMF_cropped.fits.gz
    zip_files.sort()
    result = []
    group = [zip_files[0]]
    current_id = zip_files[0].split('_')[2]

    for file in zip_files:
        id = file.split('_')[2]
        if id == current_id:
            group.append(file)
            continue
        count = len(group)

        # 出现不相等，说明一组已经便利完毕，处理group，并清空
        # 保留group中时间中间的文件，其他文件排除
        result.append(group[count // 2])
        group = [file]
        current_id = id

    if len(group) > 0:  # 处理最后一组
        count = len(group)
        result.append(group[count // 2])

    return result


def dophot(file):
    with fits_open(file, mode='update') as hdul:
        headers = hdul[0].header

        wcs = WCS(headers)
        # ra, dec = wcs.all_pix2world(512, 512, 0)
        x, y = wcs.all_world2pix(headers['T_RA'], headers['T_DEC'], 0)
        x = x.item()
        y = y.item()
        headers['T_X'] = x
        headers['T_Y'] = y

        data = hdul[0].data
        pos = np.transpose((x, y))
        # 使用2D高斯拟合计算FWHM
        cutout = Cutout2D(data, position=pos, size=(9, 9))
        cutout_data = cutout.data

        # fwhm = fit_fwhm(cutout_data)
        fwhm = 1
        # 使用半高全宽FWHM的一半作为光圈直径，计算信噪比
        aperture_radius = fwhm * 2

        apertures = CircularAperture(pos, r=aperture_radius)
        apertures_stats = ApertureStats(data, apertures)
        annulus_apertures = CircularAnnulus(pos, r_in=aperture_radius + 2, r_out=aperture_radius + 5)
        annulus_apertures_stats = ApertureStats(data, annulus_apertures)
        # bkg_value = annulus_apertures_stats.mean
        bkg_value = annulus_apertures_stats.median

        total_bkg = bkg_value * apertures.area
        noise = annulus_apertures_stats.std * np.sqrt(apertures.area)

        phot_table = aperture_photometry(data, apertures)
        flux = phot_table['aperture_sum'] - total_bkg
        flux = flux[0]  # only one target
        if flux < 0:
            flux = 0
        snr = flux / noise
        phot_table['total_bkg'] = total_bkg
        phot_table['flux'] = flux
        phot_table['bk_std'] = annulus_apertures_stats.std
        phot_table['flux_error'] = annulus_apertures_stats.std * np.sqrt(apertures.area)
        phot_table['snr'] = snr
        phot_table['fwhm'] = apertures_stats.fwhm

        headers['T_SNR'] = snr
        headers['T_APER'] = (apertures.r, f'annulus_in={annulus_apertures.r_in}, annulus_out={annulus_apertures.r_out}')
        hdul.flush()
        hdul.close()

    return x, y, snr


def astrometry(selected_file_list, original_plan_file):
    """
    解压选中的文件，增加目标位置，并添加天文定位信息
    处理以后的文件在output目录下，后缀_solved.fits
    :param selected_file_list:
    :param original_plan_file:
    :return:
    """
    if len(selected_file_list) < 1:
        return

    with open(original_plan_file) as f:
        text = '[' + f.read()[:-2] + ']'
        plan = json.loads(text)

    base_dir = path.dirname(selected_file_list[0])
    output_dir = path.join(base_dir, 'output')
    stat_f = open(path.join(output_dir, 'stat.csv'), 'w')
    stat_f.write('File, Target_ID, RA, DEC, X, Y, SNR\n')
    for file in selected_file_list:
        with fits_open(file) as hdul:
            headers = hdul[0].header

            id = file.split('_')[2]
            tid = None
            for item in plan:
                if item['target'].endswith(id[1:]):  # 第一个字符是策略编号，后面的是目标编号
                    tid = item['target']
                    ra = item['RA']
                    dec = item['Dec']
                    break
            if tid is not None:
                headers['T_ID'] = tid
                headers['T_RA'] = ra
                headers['T_DEC'] = dec
            else:
                print('Target not found in plan file:', file)
                continue

            base_name = path.basename(file)
            selected_subfix = '_selected.fits'
            unzip_name = base_name.replace('_DFLYMF_cropped.fits.gz', selected_subfix)
            unzip_fits = path.join(base_dir, unzip_name)

            solved_name = unzip_name.replace(selected_subfix, '_solved.fits')
            solved_file = path.join(output_dir, solved_name)

            if not path.exists(solved_file):  # 如果还未处理过，则进行定位处理，否则只进行测光
                fits_save(unzip_fits, hdul[0].data, header=headers, overwrite=True)
                # 完成天文定位，输出文件在output目录下，后缀_solved.fits

                cmd = (f'solve-field --no-plots --corr none --index-xyls none --match none --rdls none --solved none '
                       f' --depth 60,100 --overwrite --dir {output_dir} --new-fits {solved_file} -3 {ra} -4 {dec} '
                       f' -5 1.0 --scale-units arcsecperpix --scale-low 8.0 --scale-high 9.4 {unzip_fits} ')

                # cmd = cmd_pattern.format(output_dir=output_dir, solved_name=solved_name, ra=ra, dec=dec, unzip_fits=unzip_fits)
                print(cmd)
                exit_status = system(cmd)
                # print(exit_status)
                if exit_status != 0:  # solve-field failed
                    print('Error:', base_name)
                    continue

                if not path.exists(solved_file):  # solve-field failed
                    print('Solved file not found:', solved_file)
                    stat_f.write(f'{solved_name}, {tid}, {ra}, {dec}, {x}, {y}, {0.0}\n')
                    continue

                # remove unzip file
                remove(unzip_fits)
                # remove wcs file in output directory
                wcs_file = path.join(output_dir, unzip_name.replace('.fits', '.wcs'))
                remove(wcs_file)
                # remove axy file in output directory
                axy_file = path.join(output_dir, unzip_name.replace('.fits', '.axy'))
                remove(axy_file)

            # 对目标进行孔径测光
            x, y, snr = dophot(solved_file)
            print(solved_name, tid, 'SNR:', snr)
            stat_f.write(f'{solved_name}, {tid}, {ra}, {dec}, {x}, {y}, {snr}\n')
            stat_f.flush()

    stat_f.close()


if __name__ == '__main__':
    #source_path = '/disk16t/xiaoguangdian/caohai/20241228/20241228-2-RL/'

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='source directory')
    args = parser.parse_args()
    source_path = args.dir
    files = select_files(source_path)
    plan_file = glob.glob(path.join(source_path, 'plan_*.csv'))[0]
    astrometry(files, plan_file)
    for f in files:
        print(f)
    print(len(files), 'Done')
