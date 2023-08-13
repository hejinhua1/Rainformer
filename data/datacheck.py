import os
import pandas as pd
import pygrib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import locale
from einops import rearrange, reduce, parse_shape
import xarray as xr
import sys
import einops
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10

# 目录路径
# directory_path = 'E:/HJHCloud/Seafile/startup/Rainformer/data/01.grib2'
directory_path = u'E:/HJHCloud/Seafile/typhoon data/bigdata/pressure level/'
output_path = u'E:/HJHCloud/Seafile/typhoon data/bigdata/data1979.npy'


if __name__ == '__main__':
    variNameDict = {0: 'Geopotential', 1: 'Specific humidity', 2: 'Temperature', 3: 'U component of wind',
                    4: 'V component of wind', 5: 'Vertical velocity'}
    reversed_variNameDict = {v: k for k, v in variNameDict.items()}
    data = np.load(output_path)
    # 将维度 [8760, 13, 5, 41, 61] 变换为 [8760, 5, 13, 41, 61]
    reshaped_data = rearrange(data, 't h w x y -> t w h x y')
    preLevelDict = {0: '50', 1: '100', 2: '150', 3: '200',
                    4: '250', 5: '300', 6: '400', 7: '500',
                    8: '600', 9: '700', 10: '850', 11: '925', 12: '1000'}
    # 打印变换后的数组形状
    print(reshaped_data.shape)
    delt1 = reshaped_data[:, 0, :, :, :] - data[:, :, 0, :, :]
    delt2 = reshaped_data[:, 1, :, :, :] - data[:, :, 1, :, :]
    delt3 = reshaped_data[:, 2, :, :, :] - data[:, :, 2, :, :]
    delt4 = reshaped_data[:, 3, :, :, :] - data[:, :, 3, :, :]
    delt5 = reshaped_data[:, 4, :, :, :] - data[:, :, 4, :, :]
    delt6 = reshaped_data[:, 5, :, :, :] - data[:, :, 5, :, :]
    print(delt1.max(), delt1.min())
    print(delt2.max(), delt2.min())
    print(delt3.max(), delt3.min())
    print(delt4.max(), delt4.min())
    print(delt5.max(), delt5.min())
    print(delt6.max(), delt6.min())
    # # TODO 画图检查
    # # 创建图形
    # # plt.figure(figsize=(8, 6))
    # # plt.contourf(data[0,0,3,:,:])  # 用于气象数据的等高线填充图
    # # plt.colorbar(label='Value')  # 添加颜色条
    # # plt.xlabel('X Label')
    # # plt.ylabel('Y Label')
    # # plt.title('Meteorological Data Visualization')
    # #
    # # # 保存图形为图片文件
    # # plt.savefig('meteorological_data.png', dpi=300, bbox_inches='tight')
    #
    #
    # # # 显示图形
    # # plt.show()
    #
    # # TODO 交换维度查看
    # variable = 5
    # plot_data = reshaped_data[0,variable,:,:,:]
    # # 创建包含13个子图的图表
    # fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    #
    # # 遍历压力层，绘制每个子图
    # for i, ax in enumerate(axes.flat):
    #     if i < 13:
    #         img = ax.imshow(plot_data[i], cmap='viridis', origin='lower')
    #         ax.set_title('Pressure Level {}'.format(preLevelDict[i]))
    #         plt.colorbar(img, ax=ax, orientation='vertical', label='Value')
    #     else:
    #         ax.axis('off')  # 如果压力层数不足13个，关闭多余的子图
    #
    # # 调整子图之间的间距
    # plt.tight_layout()
    #
    # # 保存图像为PNG格式
    # plt.savefig('{}.png'.format(variNameDict[variable]), dpi=300, bbox_inches='tight')
    #
    # plt.show()
    # print('finished')

