import os
import pandas as pd
import pygrib
import numpy as np
import locale
import logging
import datetime
import xarray as xr
import sys
from einops import rearrange, reduce, parse_shape
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# 设置日志文件路径
log_file_path = 'output_log.txt'
logging.basicConfig(filename=log_file_path, level=logging.INFO)

# 目录路径
# directory_path = 'E:/HJHCloud/Seafile/startup/Rainformer/data/01.grib2'
directory_path = u'E:/HJHCloud/Seafile/typhoon data/bigdata/pressure level/'
output_path = u'E:/HJHCloud/Seafile/typhoon data/bigdata/data.npy'


if __name__ == '__main__':

    varName = ['z', 'q', 't', 'u', 'v', 'w']
    # 遍历目录下的所有文件
    for year_path in os.listdir(directory_path):
        year = int(year_path)
        if year<2018:
            continue

        oneHourData = np.zeros((13, 6, 41, 61), dtype=np.float32)
        oneHour_oneVari_Data = np.zeros((6, 41, 61), dtype=np.float32)
        i, j = 0, 0
        big_data = []
        print('#########################开始处理{}年的数据#########################'.format(year_path))
        year_file_path = os.path.join(directory_path, year_path)
        for month_path in os.listdir(year_file_path):
            month = int(month_path[:2])
            if month < 10:
                continue
            print('##################开始处理{}月的数据##################'.format(month_path[:2]))
            if month_path.endswith(".grib2"):
                # 打开 grib2 文件
                grib2file = os.path.join(year_file_path+'/', month_path)
                grbs = pygrib.open(grib2file)

                # 遍历 grib2 文件中的每个消息
                for grb in grbs:
                    i = i+1
                    # 获取消息的数据和元数据
                    lats, lons = grb.latlons()
                    date = grb.validDate
                    # dataDate = grb.dataDate
                    # dataTime = grb.dataTime
                    # julianDay = grb.julianDay
                    year, month, day, hour = grb.year, grb.month, grb.day, grb.hour
                    if date < datetime.datetime(2018, 10, 7, 4, 0, 0):
                        continue

                    shortName = grb.shortName
                    missingValue = grb.missingValue
                    # average = grb.average
                    # scaleValuesBy = grb.scaleValuesBy
                    level = grb.level
                    variable_name = grb.parameterName
                    timestamp = date.timestamp()/3600
                    data = grb.values

                    if data.shape != (41, 61):
                        print('########{}变量{}数据维度与其他不一致########'.format(date, variable_name))
                        continue

                    # TODO 用try except 处理
                    # 异常处理
                    exists_missing = np.isin(missingValue, data)
                    if exists_missing:
                        print('########{}变量{}缺失处理########'.format(date, variable_name))
                        data[data == missingValue] = np.nan
                        # 计算非缺失值的均值
                        mean_value = np.nanmean(data)
                        # 使用均值来填充缺失值
                        data[np.isnan(data)] = mean_value
                    # 修改数组元素
                    # i_vari = int((i-1) % 6)
                    # assert varName[i_vari] == shortName, '########{}变量{}数据顺序出错########'.format(date, variable_name)
                    # oneHour_oneVari_Data[i_vari] = data.astype(np.float32) # 增加同一气压层不同变量的数据
                    # if (i%6 == 0): # 增加同一气压层下所有变量的数据
                    #     i_press = int((i/6 - 1)%13)
                    #     oneHourData[i_press] = oneHour_oneVari_Data
                    # if (i % 78 == 0):  # 增加同一气压层下所有变量的数据
                    #     big_data.append(oneHourData)

                # 关闭 grib2 文件
                grbs.close()
    #     output_path = u'F:/HJH/HJHcloud/Seafile/bigdata/data{}.npy'.format(year_path)
    #     reshaped_data = rearrange(np.array(big_data), 't h w x y -> t w h x y')
    #     np.save(output_path, reshaped_data)
    #     print(
    #         '######################### {}年数据处理已完成 #########################'.format(year_path))
    # print('########################################## 数据处理已完成 ##########################################')
