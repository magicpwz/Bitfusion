import logging
import math

# import ConfigParser
import configparser as ConfigParser
import numpy as np
import inspect

from bitfusion.src.utils.utils import ceil_a_by_b, log2, lookup_pandas_dataframe
from bitfusion.src.simulator.stats import Stats
from bitfusion.src.simulator.loop_stack import LoopStack

# old 优化函数 bitfusion 
from bitfusion.src.optimizer.optimizer_bitfusion import optimize_for_order, get_stats_fast
from bitfusion.src.simulator.accelerator_sample import Accelerator

# new 优化函数 coarse
# from bitfusion.src.optimizer.optimizer_coarse import optimize_for_order, get_stats_fast
# from bitfusion.src.simulator.accelerator_coarse import Accelerator

# new fine_grain
# from bitfusion.src.optimizer.optimizer_fine_grain import optimize_for_order, get_stats_fast
# from bitfusion.src.simulator.accelerator_fine_grain import Accelerator




from bitfusion.src.simulator.energy import EnergyTuple

from bitfusion.sram.cacti_sweep import CactiSweep
import os
import pandas
import sys
import random


from dnnweaver2.tensorOps.cnn import Convolution, MatMul


class Simulator(object):
    """
    Simulator class
    """

    def __init__(self, config_file="conf.ini", verbose=False, energy_costs=None):
        # custom energy cost
        self.energy_costs = energy_costs

        self.config_file = config_file

        self.config = ConfigParser.ConfigParser()
        self.config.read(config_file)

        # 三个数的列表[a,1,c]
        # a?? c??
        systolic_dim = [
            self.config.getint("accelerator", "a"),
            1,
            self.config.getint("accelerator", "c"),
        ]

        # verbose是否进行调试
        # logging.DEBUG 级别用于输出详细的调试信息
        # logging.INFO 级别用于输出一般的信息

        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        # logging.basicConfig(level=log_level)

        self.logger = logging.getLogger("{}.{}".format(__name__, "Simulator"))
        self.logger.setLevel(log_level)
        self.logger.debug("Creating Simulator Object")

        self.logger.debug("Systolic Array dimentions: {}".format(systolic_dim))

        # if_width: Memory Interface Bit-Width

        mem_if_width = self.config.getint("system", "if_width")

        self.logger.debug("Memory Interface Bit-Width: {}-bits".format(mem_if_width))

        # 运算中的最高最低精度设置
        # high_prec -> High Precision 即一边最大支持输入8bit数据,占4个小块(最小为2bit时)
        # low_prec -> Low Precision 这里可以定义每个小块的精度

        pmax = self.config.getint("accelerator", "high_prec")
        pmin = self.config.getint("accelerator", "low_prec")

        self.logger.debug("High Precision: {}-bits".format(pmax))
        self.logger.debug("Low Precision: {}-bits".format(pmin))

        # Using half the size assuming double buffering
        sram = {}

        # Act_SRAM -> Activation SRAM size
        sram["act"] = self.config.getint("accelerator", "Act_SRAM")
        self.logger.debug("Activation SRAM size: {:,} Bytes".format(sram["act"]))

        # Wgt_SRAM -> Weight SRAM size
        sram["wgt"] = self.config.getint("accelerator", "Wgt_SRAM")
        self.logger.debug("Weight SRAM size: {:,} Bytes".format(sram["wgt"]))

        # Out_SRAM -> Output SRAM size
        sram["out"] = self.config.getint("accelerator", "Out_SRAM")
        self.logger.debug("Output SRAM size: {:,} Bytes".format(sram["out"]))

        # frequency -> Frequency 时钟频率 Hz
        # 500,000,000 Hz (0.5 GHz)
        frequency = self.config.getint("accelerator", "frequency")
        self.logger.debug("Frequency: {:,} Hz".format(frequency))

        # 单边 Fusion Units
        hp_peak_throughput = systolic_dim[0] * systolic_dim[1] * systolic_dim[2]

        # 内部小块 BitBricks
        peak_throughput = hp_peak_throughput * (int(pmax / pmin) ** 2)

        # 那么这个bb的内部融合配置呢？？

        # 数据吞吐量
        self.logger.debug(
            "Lowest  precision: Peak Throughput: {:,} Ops/cycle".format(peak_throughput)
        )
        self.logger.debug(
            "Highest precision: Peak Throughput: {:,} Ops/cycle".format(
                hp_peak_throughput
            )
        )

        N = systolic_dim[0]

        # 暂定为标记
        beta = systolic_dim[1]

        M = systolic_dim[2]

        assert beta == 1

        # 加速器设计
        # init
        self.accelerator = Accelerator(N, M, pmax, pmin, sram, mem_if_width, frequency)

        ##################################################
        # Get stats for SRAM
        frequency = self.accelerator.frequency

        # 45nm
        tech_node = 45
        # sram_csv 废弃变量
        sram_csv = "hardware_sweep/sram_results.csv"

        sram_opt_dict = {"technology (u)": tech_node * 1.0e-3}

        dir_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../sram"
        )

        """CACTI is an analytical tool that takes a set of cache/memory para-
        meters as input and calculates its access time, power, cycle time, and area.
        https://github.com/HewlettPackard/cacti
        """
        # 输入默认设置 + 工艺
        self.sram_obj = CactiSweep(
            bin_file=os.path.join(dir_path, "cacti/cacti"),
            # csv配置？？
            csv_file=os.path.join(dir_path, "cacti_sweep.csv"),
            default_json=os.path.join(dir_path, "default.json"),
            default_dict=sram_opt_dict,
        )

    def get_area(self):
        frequency = self.accelerator.frequency
        ##################################################
        N = self.accelerator.N
        M = self.accelerator.M
        pmax = self.accelerator.pmax
        pmin = self.accelerator.pmin
        wbuf_size = self.accelerator.sram["wgt"] * 8
        ibuf_size = self.accelerator.sram["act"] * 8
        obuf_size = self.accelerator.sram["out"] * 8

        wbuf_bank = N * M
        ibuf_bank = N
        obuf_bank = M

        # wbuf_bits = pmax * pmax / pmin
        # ibuf_bits = pmax * pmax / pmin

        wbuf_bits = 32
        ibuf_bits = 32

        obuf_bits = 32
        wbuf_word = ceil_a_by_b(wbuf_size, wbuf_bank * wbuf_bits)
        ibuf_word = ceil_a_by_b(ibuf_size, ibuf_bank * ibuf_bits)
        obuf_word = ceil_a_by_b(obuf_size, obuf_bank * obuf_bits)
        wbuf_bank_size = wbuf_word * wbuf_bits
        ibuf_bank_size = ibuf_word * ibuf_bits
        obuf_bank_size = obuf_word * obuf_bits

        assert wbuf_bank_size * wbuf_bank == wbuf_size
        assert ibuf_bank_size * ibuf_bank == ibuf_size
        assert obuf_bank_size * obuf_bank == obuf_size

        ##################################################
        cfg_dict = {
            "size (bytes)": wbuf_bank_size / 8.0,
            "block size (bytes)": wbuf_bits / 8.0,
            "read-write port": 0,
        }
        # wbuf area
        wbuf_data = self.sram_obj.get_data_clean(cfg_dict)
        wbuf_read_energy = float(wbuf_data["read_energy_nJ"]) / wbuf_bits
        wbuf_write_energy = float(wbuf_data["write_energy_nJ"]) / wbuf_bits
        wbuf_leak_power = float(wbuf_data["leak_power_mW"]) * wbuf_bank
        wbuf_area = float(wbuf_data["area_mm^2"]) * wbuf_bank

        self.logger.debug("WBUF :")
        self.logger.debug("\tBanks                       : {0:>8}".format(wbuf_bank))
        self.logger.debug("\tBitWidth                    : {0:>8} bits".format(wbuf_bits))
        self.logger.debug("\tWords                       : {0:>8}".format(wbuf_word))
        self.logger.debug("\tTotal Size                  : {0:>8} kBytes".format(wbuf_size / 8.0 / 1024.0))
        self.logger.debug("\tTotal Area                  : {0:>8.2f} mm^2".format(wbuf_area))
        self.logger.debug("\tLeak Energy (per clock)     : {0:>8.4f} mWatt".format(wbuf_leak_power))
        self.logger.debug("\tRead Energy                 : {0:>8.4f} pJ/bit".format(wbuf_read_energy * 1.0e3))
        self.logger.debug("\tWrite Energy                : {0:>8.4f} pJ/bit".format(wbuf_write_energy * 1.0e3))
        
        ##################################################
        cfg_dict = {
            "size (bytes)": ibuf_bank_size / 8.0,
            "block size (bytes)": ibuf_bits / 8.0,
            "read-write port": 0,
        }
        # ibuf_area
        ibuf_data = self.sram_obj.get_data_clean(cfg_dict)
        ibuf_read_energy = float(ibuf_data["read_energy_nJ"]) / ibuf_bits
        ibuf_write_energy = float(ibuf_data["write_energy_nJ"]) / ibuf_bits
        ibuf_leak_power = float(ibuf_data["leak_power_mW"]) * ibuf_bank
        ibuf_area = float(ibuf_data["area_mm^2"]) * ibuf_bank

        self.logger.debug("IBUF :")
        self.logger.debug("\tBanks                       : {0:>8}".format(ibuf_bank))
        self.logger.debug("\tBitWidth                    : {0:>8} bits".format(ibuf_bits))
        self.logger.debug("\tWords                       : {0:>8}".format(ibuf_word))
        self.logger.debug("\tTotal Size                  : {0:>8} kBytes".format(ibuf_size / 8.0 / 1024.0))
        self.logger.debug("\tTotal Area                  : {0:>8.2f} mm^2".format(ibuf_area))
        self.logger.debug("\tLeak Energy (per clock)     : {0:>8.4f} mWatt".format(ibuf_leak_power))
        self.logger.debug("\tRead Energy                 : {0:>8.4f} pJ/bit".format(ibuf_read_energy * 1.0e3))
        self.logger.debug("\tWrite Energy                : {0:>8.4f} pJ/bit".format(ibuf_write_energy * 1.0e3))
        ##################################################
        
        cfg_dict = {
            "size (bytes)": obuf_bank_size / 8.0,
            "block size (bytes)": obuf_bits / 8.0,
            "read-write port": 1,
        }

        # cfg_dict = {
        #     "size (bytes)": obuf_bank_size / 8.0,
        #     "block size (bytes)": obuf_bits / 8.0,
        #     "read-write port": 0,
        # }

        # import ipdb;ipdb.set_trace()

        # obuf_area
        obuf_data = self.sram_obj.get_data_clean(cfg_dict)
        obuf_read_energy = float(obuf_data["read_energy_nJ"]) / obuf_bits
        obuf_write_energy = float(obuf_data["write_energy_nJ"]) / obuf_bits
        obuf_leak_power = float(obuf_data["leak_power_mW"]) * obuf_bank
        obuf_area = float(obuf_data["area_mm^2"]) * obuf_bank

        self.logger.debug("OBUF :")
        self.logger.debug("\tBanks                       : {0:>8}".format(obuf_bank))
        self.logger.debug("\tBitWidth                    : {0:>8} bits".format(obuf_bits))
        self.logger.debug("\tWords                       : {0:>8}".format(obuf_word))
        self.logger.debug("\tTotal Size                  : {0:>8} kBytes".format(obuf_size / 8.0 / 1024.0))
        self.logger.debug("\tTotal Area                  : {0:>8.2f} mm^2".format(obuf_area))
        self.logger.debug("\tLeak Energy (per clock)     : {0:>8.4f} mWatt".format(obuf_leak_power))
        self.logger.debug("\tRead Energy                 : {0:>8.4f} pJ/bit".format(obuf_read_energy * 1.0e3))
        self.logger.debug("\tWrite Energy                : {0:>8.4f} pJ/bit".format(obuf_write_energy * 1.0e3))
        ##################################################
        
        # core_area
        # Get stats for systolic array
        core_csv = os.path.join("./results", "systolic_array_synth.csv")
        core_synth_data = pandas.read_csv(core_csv)

        lookup_dict = {}
        lookup_dict["Max Precision (bits)"] = pmax
        lookup_dict["Min Precision (bits)"] = pmin
        lookup_dict["N"] = N
        lookup_dict["M"] = M
        
        core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)
        if len(core_data) == 0:
            lookup_dict["N"] = 4
            lookup_dict["M"] = 4
            core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)
            assert len(core_data) == 1
            core_area = float(core_data["Area (um^2)"]) * 1.0e-6 * (N * M) / 16.0
            core_dyn_power = float(core_data["Dynamic Power (nW)"]) * (N * M) / 16.0
            core_dyn_energy = core_dyn_power / float(core_data["Frequency"])
            core_leak_power = float(core_data["Leakage Power (nW)"]) * (N * M) / 16.0
            core_leak_energy = core_leak_power / float(core_data["Frequency"])
        else:
            core_area = float(core_data["Area (um^2)"]) * 1.0e-6
            core_dyn_power = float(core_data["Dynamic Power (nW)"])
            core_dyn_energy = core_dyn_power / float(core_data["Frequency"])
            core_leak_power = float(core_data["Leakage Power (nW)"])
            core_leak_energy = core_leak_power / float(core_data["Frequency"])

        self.logger.debug("Core :")
        self.logger.debug("\tDimensions              : {0}x{1}-systolic array".format(N, M))
        self.logger.debug("\tMax-Precision           : {}".format(pmax))
        self.logger.debug("\tMin-Precision           : {}".format(pmin))
        self.logger.debug("\tLeak power              : {} (nW)".format(core_leak_energy))
        self.logger.debug("\tDynamic Energy (nJ)     : {}".format(core_dyn_energy))
        self.logger.debug("\tArea (mm^2)             : {}".format(core_area))
        ##################################################

        return core_area, wbuf_area, ibuf_area, obuf_area

    # 重要函数
    def get_energy_cost(self):
        if self.energy_costs is not None:
            return self.energy_costs

        frequency = self.accelerator.frequency
        ##################################################
        N = self.accelerator.N
        M = self.accelerator.M
        pmax = self.accelerator.pmax
        pmin = self.accelerator.pmin

        wbuf_size = self.accelerator.sram["wgt"] * 8
        ibuf_size = self.accelerator.sram["act"] * 8
        obuf_size = self.accelerator.sram["out"] * 8

        # 有图示
        # bank可以只是一个标志
        # N,M:矩阵的行列
        # bank=512表示这个bank内部有512个WBUF
        wbuf_bank = N * M
        ibuf_bank = N
        obuf_bank = M

        # 一个 WBUF or IBUF 给出的数据
        # 默认bitfusion是32
        # 精度的影响？？

        # 8 2可以
        # 8 4如果不行就使用 32 32  
        wbuf_bits = (pmax * pmax / pmin)
        ibuf_bits = (pmax * pmax / pmin)

        # wbuf_bits = 32
        # ibuf_bits = 32

        obuf_bits = 32

        # print('wbuf_size',wbuf_size)
        # print('wbuf_bank',wbuf_bank)
        # print('wbuf_bits',wbuf_bits)
        # print('wbuf_bank * wbuf_bits',wbuf_bank * wbuf_bits)

        # Bitfusion 默认wbuf_word:32
        # WORD=32 表示总大小能分32个bank

        wbuf_word = ceil_a_by_b(wbuf_size, wbuf_bank * wbuf_bits)

        ibuf_word = ceil_a_by_b(ibuf_size, ibuf_bank * ibuf_bits)
        obuf_word = ceil_a_by_b(obuf_size, obuf_bank * obuf_bits)

        # print(wbuf_word)

        # 多个bank中的单个WBUF并行时输出的值
        wbuf_bank_size = wbuf_word * wbuf_bits
        ibuf_bank_size = ibuf_word * ibuf_bits
        obuf_bank_size = obuf_word * obuf_bits

        # print(wbuf_bank_size)

        assert wbuf_bank_size * wbuf_bank == wbuf_size
        assert ibuf_bank_size * ibuf_bank == ibuf_size
        assert obuf_bank_size * obuf_bank == obuf_size

        ##################################################

        # BB_PE配置
        # size:
        # block size:
        cfg_dict = {
            "size (bytes)": wbuf_bank_size / 8.0,
            "block size (bytes)": wbuf_bits / 8.0,
            "read-write port": 0,
        }
        # cfg_dict = {
        #     "size (bytes)": wbuf_bank_size / 8.0,
        #     "block size (bytes)": wbuf_bits / 8.0,
        #     "read-write port": 1,
        # }

        # 测试数据
        # cfg_dict = {'size (bytes)': wbuf_bank_size /8., 'block size (bytes)': wbuf_bits/2., 'read-write port': 0}

        # size (bytes):单次多Bank单WBUF输出的数据Bytes,(可能是当脉动阵列，阶梯输入，每次进一个WBUF)
        # block size:单个WBUF的输出
        # get_data_clean()??

        print(cfg_dict)

        wbuf_data = self.sram_obj.get_data_clean(cfg_dict)
        # print(wbuf_data)

        wbuf_read_energy = float(wbuf_data["read_energy_nJ"]) / wbuf_bits
        wbuf_write_energy = float(wbuf_data["write_energy_nJ"]) / wbuf_bits
        # 静态功耗
        wbuf_leak_power = float(wbuf_data["leak_power_mW"]) * wbuf_bank
        wbuf_area = float(wbuf_data["area_mm^2"]) * wbuf_bank

        self.logger.debug("WBUF :")
        self.logger.debug("\tBanks                       : {0:>8}".format(wbuf_bank))
        self.logger.debug("\tBitWidth                    : {0:>8} bits".format(wbuf_bits))
        self.logger.debug("\tWords                       : {0:>8}".format(wbuf_word))
        self.logger.debug("\tTotal Size                  : {0:>8} kBytes".format(wbuf_size / 8.0 / 1024.0))
        self.logger.debug("\tTotal Area                  : {0:>8.2f} mm^2".format(wbuf_area))
        self.logger.debug("\tLeak Energy (per clock)     : {0:>8.4f} mWatt".format(wbuf_leak_power))
        self.logger.debug("\tRead Energy                 : {0:>8.4f} pJ/bit".format(wbuf_read_energy * 1.0e3))
        self.logger.debug("\tWrite Energy                : {0:>8.4f} pJ/bit".format(wbuf_write_energy * 1.0e3))
        ##################################################

        # 原始
        cfg_dict = {
            "size (bytes)": ibuf_bank_size / 8.0,
            "block size (bytes)": ibuf_bits / 8.0,
            "read-write port": 0,
        }
        print("cfg_dict", cfg_dict)
        # 测试
        # cfg_dict = {'size (bytes)': ibuf_bank_size /8., 'block size (bytes)': ibuf_bits/2., 'read-write port': 0}

        ibuf_data = self.sram_obj.get_data_clean(cfg_dict)
        ibuf_read_energy = float(ibuf_data["read_energy_nJ"]) / ibuf_bits
        ibuf_write_energy = float(ibuf_data["write_energy_nJ"]) / ibuf_bits
        ibuf_leak_power = float(ibuf_data["leak_power_mW"]) * ibuf_bank
        ibuf_area = float(ibuf_data["area_mm^2"]) * ibuf_bank

        self.logger.debug("IBUF :")
        self.logger.debug("\tBanks                       : {0:>8}".format(ibuf_bank))
        self.logger.debug("\tBitWidth                    : {0:>8} bits".format(ibuf_bits))
        self.logger.debug("\tWords                       : {0:>8}".format(ibuf_word))
        self.logger.debug("\tTotal Size                  : {0:>8} kBytes".format(ibuf_size / 8.0 / 1024.0))
        self.logger.debug("\tTotal Area                  : {0:>8.2f} mm^2".format(ibuf_area))
        self.logger.debug("\tLeak Energy (per clock)     : {0:>8.4f} mWatt".format(ibuf_leak_power))
        self.logger.debug("\tRead Energy                 : {0:>8.4f} pJ/bit".format(ibuf_read_energy * 1.0e3))
        self.logger.debug("\tWrite Energy                : {0:>8.4f} pJ/bit".format(ibuf_write_energy * 1.0e3))
        ##################################################

        # old_未经过修改 use to 16*32
        # cfg_dict = {
        #     "size (bytes)": obuf_bank_size / 8.0,
        #     "block size (bytes)": obuf_bits / 8.0,
        #     "read-write port": 1,
        # }

        # cfg_dict = {
        #     "size (bytes)": obuf_bank_size / 8.0,
        #     "block size (bytes)": obuf_bits / 8.0,
        #     "read-write port": 0,
        # }

        print("cfg_dict_obuf", cfg_dict)

        obuf_data = self.sram_obj.get_data_clean(cfg_dict)
        obuf_read_energy = float(obuf_data["read_energy_nJ"]) / obuf_bits
        obuf_write_energy = float(obuf_data["write_energy_nJ"]) / obuf_bits
        obuf_leak_power = float(obuf_data["leak_power_mW"]) * obuf_bank
        obuf_area = float(obuf_data["area_mm^2"]) * obuf_bank

        self.logger.debug("OBUF :")
        self.logger.debug("\tBanks                       : {0:>8}".format(obuf_bank))
        self.logger.debug("\tBitWidth                    : {0:>8} bits".format(obuf_bits))
        self.logger.debug("\tWords                       : {0:>8}".format(obuf_word))
        self.logger.debug("\tTotal Size                  : {0:>8} kBytes".format(obuf_size / 8.0 / 1024.0))
        self.logger.debug("\tTotal Area                  : {0:>8.2f} mm^2".format(obuf_area))
        self.logger.debug("\tLeak Energy (per clock)     : {0:>8.4f} mWatt".format(obuf_leak_power))
        self.logger.debug("\tRead Energy                 : {0:>8.4f} pJ/bit".format(obuf_read_energy * 1.0e3))
        self.logger.debug("\tWrite Energy                : {0:>8.4f} pJ/bit".format(obuf_write_energy * 1.0e3))
        ##################################################
        # Get stats for systolic array

        core_csv = os.path.join("./results", "systolic_array_synth.csv")

        core_synth_data = pandas.read_csv(core_csv)

        lookup_dict = {}
        lookup_dict["Max Precision (bits)"] = pmax
        lookup_dict["Min Precision (bits)"] = pmin

        lookup_dict["N"] = N
        lookup_dict["M"] = M

        core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)
        
        print()
        print('NM_MAX_MIN_参数适配:',len(core_data))

        if len(core_data) == 0:

            lookup_dict["N"] = 4
            lookup_dict["M"] = 4
            # 把长宽修正到4 4,最大最小精度还是得符合CSV文件
            core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)

            # 16 & 32被修正成了4 4

            print("修正后,lookup_dict", lookup_dict)
            print()

            # 过个断言,csv文件定死了最大最小精度
            assert len(core_data) == 1

            # 按照倍数进行计算 (N*M)/(4*4)
            core_area = float(core_data["Area (um^2)"]) * 1.0e-6 * (N * M) / 16.0
            core_dyn_power = float(core_data["Dynamic Power (nW)"]) * (N * M) / 16.0
            core_dyn_energy = core_dyn_power / float(core_data["Frequency"])
            core_leak_power = float(core_data["Leakage Power (nW)"]) * (N * M) / 16.0
            core_leak_energy = core_leak_power / float(core_data["Frequency"])

        else:
            core_area = float(core_data["Area (um^2)"]) * 1.0e-6
            core_dyn_power = float(core_data["Dynamic Power (nW)"])
            core_dyn_energy = core_dyn_power / float(core_data["Frequency"])
            core_leak_power = float(core_data["Leakage Power (nW)"])
            core_leak_energy = core_leak_power / float(core_data["Frequency"])

        self.logger.debug("Core :")
        self.logger.debug("\tDimensions              : {0}x{1}-systolic array".format(N, M))
        self.logger.debug("\tMax-Precision           : {}".format(pmax))
        self.logger.debug("\tMin-Precision           : {}".format(pmin))
        self.logger.debug("\tLeak power              : {} (nW)".format(core_leak_energy))
        self.logger.debug("\tDynamic Energy (nJ)     : {}".format(core_dyn_energy))
        self.logger.debug("\tArea (mm^2)             : {}".format(core_area))
        ##################################################

        energy_tuple = EnergyTuple(
            core_dyn_energy,
            wbuf_read_energy,
            wbuf_write_energy,
            ibuf_read_energy,
            ibuf_write_energy,
            obuf_read_energy,
            obuf_write_energy,
        )

        return energy_tuple

    def __str__(self):
        ret = ""
        ret += "Simulator object"
        ret += "\n"
        ret += "\tMax supported precision: {}".format(self.accelerator.pmax)
        ret += "\n"
        ret += "\tMin supported precision: {}".format(self.accelerator.pmin)
        ret += "\n"

        ret += "\tSystolic array size: {} -inputs x {} -outputs".format(
            self.accelerator.N, self.accelerator.M
        )

        ret += "\n"
        ret += "\tWbuf size: {:,} Bytes".format(self.accelerator.sram["wgt"])
        ret += "\n"
        ret += "\tIbuf size: {:,} Bytes".format(self.accelerator.sram["act"])
        ret += "\n"
        ret += "\tObuf size: {:,} Bytes".format(self.accelerator.sram["out"])
        ret += "\n"
        ret += "Double buffering enabled. Sizes of SRAM are halved"
        return ret

    def loop_estimate_stats(self, loop_instruction, verbose=False):
        """
        args:
            loop_instruction: Loops for the NN.
                index 0 = outer loop
                index -1 = inner loop
        """

        # The following loop promotes Memory accesses to improve reuse
        loop_instruction.promote_mem_ops(self.accelerator.sram)
        # get stats
        stats = loop_instruction.get_stats(self.accelerator, verbose)

        return stats

    def get_FC_cycles(self, Ni, No, iprec, wprec, batch_size=1):
        """
        Get number of cycles required for Fully-Connected Layer.

        args:
            Ni: Input neurons
            No: Output neurons
            batch_size: Batch size for FC layer

            iprec: Precision for activations (bits)
            wprec: Precision for weights (bits)

            batch_size: Batch size for the layer

        description:
            This function calls the get_conv_cycles function
        """
        total_cycles = self.get_conv_cycles(1, 1, 1, Ni, No, iprec, wprec, batch_size)

        return total_cycles

    def get_perf_factor(self, iprec, wprec):
        iprec = max(iprec, self.accelerator.pmin)
        wprec = max(wprec, self.accelerator.pmin)
        return int(self.accelerator.pmax / iprec) * int(self.accelerator.pmax / wprec)

    def get_conv_cycles(
        self, K, O, S, IC, OC, iprec, wprec, batch_size=1, im2col=False
    ):
        """
        Get number of cycles required for Convolution layer.

        description:
            This functions does an exhaustive search for finding the optimal
            Tiling and Ordering parameters
        """

        # print('im2col',im2col)

        B = batch_size
        # print('b',B)
        # 输入数据大小
        I = (O - 1) * S + K

        # We do not tile the "K" dimension and compute an entire 2-D conv at a
        # time
        # 以2为底O的对数
        # 分模块？？
        
        num_O_tiles = int(math.ceil(log2(O))) + 1

        num_IC_tiles = int(math.ceil(log2(IC))) + 1

        num_OC_tiles = (
            int(math.ceil(log2(math.ceil(float(OC) / self.accelerator.M)))) + 1
        )

        num_B_tiles = int(math.ceil(log2(B))) + 1

        self.logger.debug("Number of O Tiles: {}".format(num_O_tiles))
        self.logger.debug("Number of IC Tiles: {}".format(num_IC_tiles))
        self.logger.debug("Number of OC Tiles: {}".format(num_OC_tiles))
        self.logger.debug("Number of B Tiles: {}".format(num_B_tiles))



        # sys.exit()
        best_instructions_dict = {}
        # print('1')
        # self.get_energy_cost() 还是得解决,不然跑不下去

        # 输入数据大小 可反推
        # I = (O - 1) * S + K


        # 动态
        # 随机概率 ->针对每一个卷积 or FC 给出一个概率分配方案
        # 生成三个随机数
        # for input
        i_random1 = random.randint(1,15)
        i_random2 = random.randint(1,20)
        i_random3 = random.randint(1,5)


        # 随机概率 ->针对每一个卷积 or FC 给出一个概率分配方案
        # 生成三个随机数
        # for weight
        w_random1 = random.randint(1,15)
        w_random2 = random.randint(1,20)
        w_random3 = random.randint(1,5)


        # 计算随机数的和
        i_total_sum = i_random1 + i_random2 + i_random3
        w_total_sum = w_random1 + w_random2 + w_random3

        # 计算每个变量的值 input
        i_low = i_random1 / i_total_sum
        i_mid = i_random2 / i_total_sum
        i_high = i_random3 / i_total_sum

        # 计算每个变量的值 weight
        w_low = w_random1 / w_total_sum
        w_mid = w_random2 / w_total_sum
        w_high = w_random3 / w_total_sum


        avg_iprec = 2 * i_low + 4 * i_mid + 8 * i_high
        avg_wprec = 2 * w_low + 4 * w_mid + 8 * w_high
        

    
        function_source_file = inspect.getsourcefile(optimize_for_order)
        file_name = os.path.basename(function_source_file)
        
        print(file_name)
        # import ipdb;ipdb.set_trace()

        if file_name == 'optimizer_bitfusion.py':
            conv_params = (
            self.accelerator,
            K,
            O,
            S,
            IC,
            OC,
            B,
            iprec,
            wprec,
            im2col,
            self.get_energy_cost(),
        )
        else:

            iprec = avg_iprec
            wprec = avg_wprec

            conv_params = (
                self.accelerator,
                K,
                O,
                S,
                IC,
                OC,
                B,
                iprec,
                wprec,
                im2col,
                self.get_energy_cost(),
                i_low,
                i_mid,
                i_high,
                w_low,
                w_mid,
                w_high
            )
        

        # TODO 分配相关？？
        # 具体优化细节
        

        best_instructions, best_tiling, best_order = optimize_for_order(conv_params) # 找到最优的 tiling 和 order 的策略
        # best_instructions 操作

        # import ipdb; ipdb.set_trace()
        stats = get_stats_fast(conv_params, best_tiling, best_order,verbose=False) # 根据相应的 调度策略来得到最终的结果



        # 这三个参数不是很懂
        print("best_instructions", best_instructions)
        print("best_tiling", best_tiling)
        # best_order: ('B/b', 'OC/oc', 'OW/ow', 'IC/ic', 'OH/oh')
        print("best_order", best_order)

        # import ipdb; ipdb.set_trace()

        # sys.exit()

        # 每一层的内存读写！！
        # act_reads = stats.reads["act"]
        # wgt_reads = stats.reads["wgt"]
        # out_reads = stats.reads["out"]
        # dram_reads = stats.reads["dram"]
        # out_writes = stats.writes["out"]
        # dram_writes = stats.writes["dram"]

        # 看cycle
        best_cycles = stats.total_cycles



        # # #fc层测试
        # # if( K == 1 and O == 1 and S == 1  ):
        # #     print('cycle',best_cycles)
        # #     # 优化后的参数
            
        # num_b, b = best_tiling["B/b"]
        # num_ow, ow = best_tiling["OW/ow"]
        # num_oh, oh = best_tiling["OH/oh"]
        # num_ic, ic = best_tiling["IC/ic"]
        # num_oc, oc = best_tiling["OC/oc"]
        # # 优化后的tiling
        # num_tiles = num_b * num_ow * num_oh * num_ic * num_oc

        # print('num_b * num_ow * num_oh * num_ic * num_oc',
            #   num_b , num_ow , num_oh , num_ic , num_oc)
        
        #     kw = kh = K
        #     print('num_tiles',num_tiles)
        #     print('stats.mem_stall_cycles',stats.mem_stall_cycles)
        #     print('stats.compute_cycles',best_cycles - stats.mem_stall_cycles)



        #     print('test:',ic, oc, ow, oh, b, kw, kh, iprec, wprec, im2col)

            # sys.exit() 
            # 为什么这里会有 sys.exit() 直接退出系统了，那后续的就不会执行了？ 
        

        num_ops = O * O * K * K * IC * OC * B # IC input-channel  OC: output-channel O: output width/length K: input width/length B: batch-size 

        # self.logger.debug('Best Operations: {}'.format(best_operations))

        self.logger.debug("Conv Layer")
        self.logger.debug("Num of ops: {}".format(num_ops))
        self.logger.debug("Kernel Size: {}x{}x{}x{}".format(K, K, IC, OC))
        self.logger.debug("Output Size: {}x{}x{}".format(O, O, OC))
        self.logger.debug("Stride Size: {}x{}".format(S, S))
        # 输入数据大小
        self.logger.debug("Input  Size: {}x{}x{}".format(I, I, IC))

        self.logger.debug("Max Precision: {}".format(self.accelerator.pmax))
        self.logger.debug("Min Precision: {}".format(self.accelerator.pmin))

        self.logger.debug("Activation Precision: {}".format(iprec))
        self.logger.debug("Weight Precision: {}".format(wprec))
        self.logger.debug(
            "Performance Factor: {}".format(self.get_perf_factor(iprec, wprec))
        )

        self.logger.debug("Total Cycles: {:,}".format(best_cycles))

        # 执行完这个layer需要的cycle
        print("Total Cycles: {:,}".format(best_cycles))
        # sys.exit()

        cycles_per_batch = ceil_a_by_b(best_cycles, B)
        self.logger.debug("Total Cycles per batch: {:,}".format(cycles_per_batch))

        ops_per_cycle = float(num_ops) / best_cycles
        self.logger.debug("Ops/Cycle: {:,.2f}".format(ops_per_cycle))

        # PE
        ops_per_cycle_per_pe = float(ops_per_cycle) / (
            self.accelerator.N * self.accelerator.M
        )

        # 0.781 0.9998
        self.logger.debug("Ops/Cycle/PE: {:,.4}".format(ops_per_cycle_per_pe))
        # sys.exit()
        # print("Ops/Cycle/PE: {:,.4}".format(ops_per_cycle_per_pe))

        return stats, best_instructions

    def get_cycles(self, op, im2col=False):
        # 卷积or全连接
        if isinstance(op, Convolution):
            # print('1')

            # B: batch_size
            # I/O: image宽高
            # IC: 输入通道

            # op.data <- from i=get_tensor()

            B, I, _, IC = op.data.shape
            # print("op.data.shape", op.data.shape)

            # O,K 也能算出数据大小
            _, O, _, OC = op.output_tensors.shape
            # print("op.output_tensors.shape", op.output_tensors.shape)
            # 中断
            # sys.exit()
            _, K, _, _ = op.weights.shape
            _, S, _, _ = op.stride

            # iprec: Precision for activations (bits)
            # wprec: Precision for weights (bits)

            # 量化精度
            iprec = op.data.dtype.bits
            wprec = op.weights.dtype.bits

            # print("op.data.op", op.data.op)

            # 这个判断实际上没有起作用,op.data.op就不存在 or 人家使用的dataware和我们的不同
            # 单纯做了个标记
            # 全部用im2col
            # if op.data.op is None:
            #     im2col = True  # im2col for first layer
            # else:
            #     im2col = False

            im2col = True

            return self.get_conv_cycles(K, O, S, IC, OC, iprec, wprec, B, im2col)

        elif isinstance(op, MatMul):
            B = op.data.shape[0]
            OC, IC = op.weights.shape


            # print('IC_for_FC',IC)
            # print('B_for_FC',B)
            # print('OC_for_FC',OC)

            # sys.exit()

            iprec = op.data.dtype.bits
            wprec = op.weights.dtype.bits
            return self.get_FC_cycles(IC, OC, iprec, wprec, batch_size=B)
