from peakaboo import finding
import pandas as pd


def test_output():
    matx_filename = '20180418_twogaussian_spectralshfit.txt'
    datanm, datatime, dataz_matx = finding.loaddata(matx_filename)
    true_df, smooth_df = finding.earth_peak_matrix(
        datanm, dataz_matx, 0.1, 0, 10)
    if isinstance(true_df, pd.DataFrame):
        pass
    else:
        raise Exception('Bad type', 'Not a dataframe')
    if isinstance(smooth_df, pd.DataFrame):
        pass
    else:
        raise Exception('Bad type', 'Not a dataframe')
    return


def test_peak_indexs():
    matx_filename = '20180418_twogaussian_spectralshfit.txt'
    datanm, datatime, dataz_matx = finding.loaddata(matx_filename)
    true_df, smooth_df = finding.earth_peak_matrix(
        datanm, dataz_matx, 0.1, 0, 10)

    i = 0
    assert true_df.iloc[i][i] == finding.findpeak(dataz_matx[:, i], 0.1, 50)[
        i], 'cannot get right true peak indexs'
    # check if we can get right smooth value from noisy dataset
    i = 0
    smoothz_matx = finding.earth_Smoothing(datanm, dataz_matx[:, i], 0.1)

    assert smooth_df.iloc[i][i] == finding.findpeak(
        smoothz_matx, 0.1, 50)[i], 'cannot get right smooth peak indexs'
    return


def test_peak_height_fwhm():
    matx_filename = '20180418_twogaussian_spectralshfit.txt'
    datanm, datatime, dataz_matx = finding.loaddata(matx_filename)
    noisez_matx, smooth_matx = finding.earth_smooth_matrix(
        datanm, dataz_matx, 0.1)
    peak_idx_df, peak_height_df, peak_fwhm_df = finding.peak_matrix(
        datanm, smooth_matx, 0.00, 50)

    i = 0
    data_timeslice = smooth_matx.values[:, i]
    peak_idx = finding.findpeak(data_timeslice, 0, 50).tolist()
    peak_height, peak_fwhm = finding.peakchar(datanm, data_timeslice, peak_idx)
    assert peak_height_df.iloc[i][i] == peak_height[i], 'cannot get right peak height value'
    # check if we can get right smooth value from noisy dataset
    assert peak_fwhm_df.iloc[i][i] == peak_fwhm[i], 'cannot get right smooth width value'
    return
