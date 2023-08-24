import pandas
import configparser as ConfigParser
import os
import numpy as np
# %load_ext autoreload
# %autoreload 2
from bitfusion.graph_plot.barchart import BarChart

# %matplotlib inline
import matplotlib

import warnings
warnings.filterwarnings('ignore')

import bitfusion.src.benchmarks.benchmarks as benchmarks
from bitfusion.src.simulator.stats import Stats
from bitfusion.src.simulator.simulator import Simulator
from bitfusion.src.sweep.sweep import SimulatorSweep, check_pandas_or_run
from bitfusion.src.utils.utils import *
from bitfusion.src.optimizer.optimizer import optimize_for_order, get_stats_fast


# ant配置
# batch_size = 64
batch_size = 1

# 默认配置
# batch_size = 16
# batch_size = 32

results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

fig_dir = './fig'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# 初始化+能耗仿真
# # BitFusion configuration file
# 原始样例
# config_file = 'bf_e_conf.ini'
# 测试样例
config_file = 'bf_test_conf.ini'

# ANT configuration file
# config_file = 'conf_ant.ini'

# Create simulator object
verbose = False
# verbose 调试模式启动
# verbose = True

#配置参数获取
config_ini = ConfigParser.ConfigParser()
config_ini.read(config_file)

bf_e_sim = Simulator(config_file, verbose)

# 得把这个解决
bf_e_energy_costs = bf_e_sim.get_energy_cost()

print(bf_e_sim)

energy_tuple = bf_e_energy_costs
print('')
print('*'*50)
print(energy_tuple)

# csv表的列头
sim_sweep_columns = ['N', 'M',
        'Max Precision (bits)', 'Min Precision (bits)',
        'Network', 'Layer',
        'Cycles', 'Memory wait cycles',
        'WBUF Read', 'WBUF Write',
        'OBUF Read', 'OBUF Write',
        'IBUF Read', 'IBUF Write',
        'DRAM Read', 'DRAM Write',
        'Bandwidth (bits/cycle)',
        'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
        'Batch size',
        'avg_iprec','avg_wprec']

# bf_e_sim_sweep_csv = os.path.join(results_dir, 'bitfusion-eyeriss-sim-sweep.csv')

bf_e_sim_sweep_csv = os.path.join(results_dir, '16*32_di_dw_bitfusion_'
                                  + str(config_ini.getint('system','if_width'))+
                                  '_sim_sweep.csv')

# bf_e_sim_sweep_csv = os.path.join(results_dir, '16*32_Nomal_bitfusion_'
#                                   + str(config_ini.getint('system','if_width'))+
#                                   '_sim_sweep.csv')


#对文件的存在进行判断
# if os.path.exists(bf_e_sim_sweep_csv):
#     bf_e_sim_sweep_df = pandas.read_csv(bf_e_sim_sweep_csv)
# else:
#     #0行21列 空表
#     bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
# print('Got BitFusion Eyeriss, Numbers')

# 直接生成新表
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)



print('bb:',batch_size)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df,
        bf_e_sim_sweep_csv, batch_size=batch_size)

# 'groupby()'方法对 'Network' 列进行分组，并使用 agg() 方法对每个组进行聚合操作，将聚合结果求和。
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)

#面积区域的放置存在问题,仿真的参数表需要重新设置

area_stats = bf_e_sim.get_area()

print('area_stats',area_stats)

#此时已经得到了Bit Fusion的 能耗+Benchmarks的次数+面积


