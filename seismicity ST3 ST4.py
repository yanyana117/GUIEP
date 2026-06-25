import numpy as np
import pandas as pd
import math

"""
- Xingran Huang
- Earthquake seismicity throughout 1900-2020 in US
    1900-04-30 ~ 2020-12-14 期间任何时间都可计算
    如果要计算小范围,需自行去网站下载小范围数据,再放进去使用计算.
- Keep updated:
    1. 震级 vs 时间序列 m-t 图 (短期)
    2. log t vs log (delta t): 强震 余震 之间的时间; 主震t=0; 同一地点, 应力关系20~30 公里范围内; 半径在50km范围内,时间图
    3. 细化到地区
    4. 细化时间数列
    5. 根据经纬度算出 ST4
- Introduction:
    1. Download Visual Studio Code in laptop:
    https://code.visualstudio.com/download
    2. 安装教程:
    https://docs.microsoft.com/zh-cn/learn/modules/python-install-vscode/
    3. 代码:
    data = pd.read_csv('/Users/yanvia/Desktop/GUIEP/S_python/query.csv')
    ('xxx/xxx/....')更改为自己电脑文件的储存路径, csv文件需要和python文件共同在同一个文件夹下
    4. 模板块下载:
    numpy,pandas,math:
                    pip install numpy
                    pip install pandas
                    pip install math
- 存储路径问题:
    更改"data = pd.read_csv('/Users/yanvia/Desktop/GUIEP/Part1_Seismicity/S_python/query.csv')"代码,
    1. 把数据库文件"query.csv"和本脚本,放到同一文件夹下
    2. 右击"query.csv"文件,点击"属性(R)"或者"get info", 找到位置/where, 复制.
    3. 把此脚本的代码 替换成 刚才复制的位置内容: data = pd.read_csv('xxxx/xxxx/query.csv')
设置为本地文件储存路径.
- 数据库:
    打开CSV格式文件,替换第一列"T"为"-", 然后确认第一列第一行的名称为"time"
- For ST4:
    # latitude: phi (phi_i, phi_j)
    # longitude: lambda (lambda_i, lambda_j)
    # inpout earthquake_i start time & earthquake_j end time
- Reference :
https://earthquake.usgs.gov/earthquakes/map/?extent=11.60919,-144.22852&extent=58.03137,-45.79102
"""


def DateSplit(df, col):
    temp_df = df[col].str.split('-', expand=True)
    temp_df.columns = ["year", "month", "day", "hour"]
    df = pd.concat([df, temp_df], axis=1)
    df = df.drop("time", axis=1)
    df['date'] = df['year'] + '-' + df['month'] + '-' + df['day']
    df['Day Time'] = df['year'] + df['month'] + df['day']
    return df


data = pd.read_csv('/Users/yanvia/Desktop/GUIEP/Part1_Seismicity/S_python/query.csv')
data= DateSplit(df=data,col='time')
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')


print('Date format: year-month-day,such as: 1900-04-30.')
time_start = input('Enter the starting time:')
time_end = input('Enter the ending time:')
Time_period = data[time_start:time_end]
N = len(Time_period)

print("{:-^50s}".format("ST3"))
# --------------------ST 3-----------------------

M_max = Time_period.loc[Time_period["mag"]==Time_period.loc[:,"mag"].max(),"mag"]
M_max = list(Time_period.loc[Time_period["mag"]==Time_period.loc[:,"mag"].max(),"mag"])
Mi = list(Time_period["mag"])

sum=0
for Mi in list(Time_period["mag"]):
    sum += 10**(1.5*Mi)

Seismicity = 1.17 * math.log10(N+1) + 0.29 * math.log10( (1/N) * sum ) + 0.15 * M_max[0]

print('ST3 = {}'.format(Seismicity))
print('Mi = {}'.format(list(Time_period["mag"])))
print('N = {} '.format(N))
print('M Max = {}'.format(M_max[0]))
print("{:-^50s}".format("ST4"))

# -------------------ST 4------------------------

def ST4(phi_i,lambda_i,phi_j,lambda_j,n):

    # A = round(math.cos(np.deg2rad(phi_i)),2)
    # B = round(math.cos(np.deg2rad(phi_j)),2)
    # C = round(math.cos(np.deg2rad(lambda_i - lambda_j)),2)
    # D = round(math.sin(np.deg2rad(phi_i)),2)
    # E = round(math.sin(np.deg2rad(phi_j)),2)

    A = math.cos(np.deg2rad(phi_i))
    B = math.cos(np.deg2rad(phi_j))
    C = math.cos(np.deg2rad(lambda_i - lambda_j))
    D = math.sin(np.deg2rad(phi_i))
    E = math.sin(np.deg2rad(phi_j))
    theta_ij = 1/math.sqrt(2) * ((1 - A*B*C - D*E))**(1/2)

    R0 = 6370 # km
    d_ij = 2 * R0 * np.arcsin(np.deg2rad(theta_ij))
    d = 1/(n*(n-1)) * d_ij
    k = 0.01 # /km
    s4 = 0.375*10**(-k*d)
    return s4


phi = list(Time_period["latitude"])
lam = list(Time_period["longitude"])
s4 = ST4(phi_i=phi[0],lambda_i=lam[0],phi_j=phi[-1],lambda_j=lam[-1],n=N)

print('ST4 = {}\n'.format(s4))
print('Starting earthquake: {}'.format(time_start))
print('Latitude phi_i = {} in deg'.format(phi[0]))
print('Longitude lambda_i = {} in deg\n'.format(lam[0]))
print('Ending earthquake: {}'.format(time_end))
print('Latitude phi_j = {} in deg'.format(phi[-1]))
print('Longitude lambda_j = {} in deg'.format(lam[-1]))
