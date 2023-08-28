from bitfusion.src.utils.utils import ceil_a_by_b, log2
from bitfusion.src.simulator.stats import Stats
import sys
import numpy as np

class Accelerator(object):
    # 不涉及具体架构的
    # 传输层次需要的cycle??
    def __init__(self, N, M, pmax, pmin, sram, mem_if_width, frequency):
        """
        accelerator object
        """
        self.N = N
        self.M = M

        # SRAM
        self.sram = sram

        # 内存带宽？
        # 参数 in ini
        self.mem_if_width = mem_if_width

        self.frequency = frequency
        self.pmax = pmax
        self.pmin = pmin

    def get_mem_read_cycles(self, dst, size):
        
        """
        Read instruction
        args:
            src_idx: index of source address
            dst: destination address
            size: size of data in bits
        """
        # 计算读取需要消耗的cycle数
        # size/带宽 -> 向上取整

        return ceil_a_by_b(size, self.mem_if_width)

    def get_mem_write_cycles(self, src, size):
        """
        Write instruction
        args:
            src_idx: index of source address
            src: destination address
            size: size of data in bits
        """
        return ceil_a_by_b(size, self.mem_if_width)

    def get_compute_stats(self, ic, oc, ow, oh, b, kw, kh, iprec, wprec, im2col=False):
        """
        Compute instruction
        args:
            ic: Input Channels
            oc: Output Channels
            ow: Output Width
            oh: Output Height
            kw: Output Height
            kh: Output Height
            b: Batch Size
            im2col: boolean. If true, we assume the cpu does im2col. Otherwise,
                    we do convolutions channel-wise
        """
        compute_stats = Stats()
        compute_stats.total_cycles = self.get_compute_cycles(
            ic, oc, ow, oh, b, kw, kh, iprec, wprec, im2col
        )
        return compute_stats

    def get_perf_factor(self, iprec, wprec):
        iprec = max(iprec, self.pmin)
        wprec = max(wprec, self.pmin)

        return int(self.pmax / iprec) * int(self.pmax / wprec)

    # ant
    # def get_perf_factor(self, prec):
    #     prec = max(prec, self.pmin)
    #     return int(self.pmax / prec)

    def generate_random_matrix(self,OW, OH, B, KW, KH, IC, row_sums):
        col_sum = KW * KH * IC
        num_columns = OW * OH * B
        random_matrix = np.random.rand(9, num_columns)

        scaled_matrix = random_matrix / random_matrix.sum(axis=0) * col_sum

        row_scale_factors = np.array(row_sums) / scaled_matrix.sum(axis=1)
        final_matrix = scaled_matrix * row_scale_factors[:, np.newaxis]

        final_matrix = np.ceil(final_matrix).astype(int)


        # 列差值修正
        gaps_col = col_sum - final_matrix.sum(axis=0)

        for x in range(OW * OH * B):
            # 均摊作差
            count = abs(gaps_col[x])
            # 单次减多少
            cut = round(count / 6) + 1

            for y in range(9):
        
                if gaps_col[x] > 0:
                    final_matrix[y][x] = final_matrix[y][x] + gaps_col[x]
                    break
                else:

                    if (final_matrix[y][x] - cut >= 0) and (count - cut>=0):
                        final_matrix[y][x] -= cut
                        count -= cut

                    elif (final_matrix[y][x] - count >= 0) and (count - cut <= 0):
                        final_matrix[y][x] -= count
                        count = 0
                    
                    if count == 0:
                        break


            # final_matrix.sum(axis=0)


        # import ipdb;ipdb.set_trace()

        return final_matrix


    def get_compute_cycles(self, ic, oc, ow, oh, b, kw, kh, iprec, wprec, im2col=False,i_low=0,i_mid=0,i_high=0,w_low=0,w_mid=0,w_high=0):
        """
        Compute instruction
        args:
            ic: Input Channels
            oc: Output Channels
            ow: Output Width
            oh: Output Height
            kw: Output Height
            kh: Output Height
            b: Batch Size
            im2col: boolean. If true, we assume the cpu does im2col. Otherwise,
                    we do convolutions channel-wise
            low: iprec 为2bit的概率
            mid: iprec 为4bit的概率
            high: iprec 为8bit的概率
        """
        # import ipdb;ipdb.set_trace()
        im2col = False

        conv_promote = {}

        conv_promote["low_low"] = {"iprec":2,"wprec":2}
        conv_promote["low_mid"] = {"iprec":2,"wprec":4}
        conv_promote["low_high"] = {"iprec":2,"wprec":8}

        conv_promote["mid_low"] = {"iprec":4,"wprec":2}
        conv_promote["mid_mid"] = {"iprec":4,"wprec":4}
        conv_promote["mid_high"] = {"iprec":4,"wprec":8}

        conv_promote["high_low"] = {"iprec":8,"wprec":2}
        conv_promote["high_mid"] = {"iprec":8,"wprec":4}
        conv_promote["high_high"] = {"iprec":8,"wprec":8}


        overhead = 0
        if (i_low==0 and i_mid==0 and i_high==0 and w_low==0 and w_mid==0 and w_high==0):
            if im2col:
                ni = kw * kh * ic
                no = oc
                batch = b * oh * ow

                # 这里和ant有区别
                # 有M N说明跟矩阵大小有关系
                compute_cycles = (
                    batch
                    * ceil_a_by_b(no, self.M)
                    * (
                        # 这个加个选择，就可以实现内部动态精度调配
                        ceil_a_by_b(ni, self.N * self.get_perf_factor(iprec, wprec))
                        + overhead
                    )
                )

                
                # ant
                # compute_cycles = batch * ceil_a_by_b(no, self.M * self.get_perf_factor(wprec)) * \
                #         (ceil_a_by_b(ni, self.N * self.get_perf_factor(iprec)))

            else:
                compute_cycles = (
                    b
                    * ceil_a_by_b(oc, self.M)
                    # 顺序执行，一个卷积算完再算下一个
                    * ow
                    * oh
                    * kw
                    * kh
                    * (
                        # 类似整体模块资源划分函数 
                        # TODO 进行计算资源划分修改
                        ceil_a_by_b(ic, self.N * self.get_perf_factor(iprec, wprec))
                        + overhead
                    )
                )
        else:
            if im2col:
                ni = kw * kh * ic
                no = oc
                batch = b * oh * ow

                # old
                # # 这里和ant有区别
                # # 有M N说明跟矩阵大小有关系
                # compute_cycles = (
                #     batch
                #     * ceil_a_by_b(no, self.M)
                #     * (
                #         # 这个加个选择，就可以实现内部动态精度调配
                #         ceil_a_by_b(ni, self.N * self.get_perf_factor(iprec, wprec))
                #         + overhead
                #     )
                # )


                # 生成随机数组
                # 9行 -> 9种可能性
                # ow*oh*b 列
                # 每一列的和为kw * kh * ic 表示单次一共需要进行计算的块

                conv_set = {}

                conv_nums = batch * ni

                conv_set['low_low'] = round(conv_nums * i_low * w_low)
                conv_set['low_mid'] = round(conv_nums * i_low * w_mid)
                conv_set['low_high'] = round(conv_nums * i_low * w_high)

                conv_set['mid_low'] = round(conv_nums * i_mid * w_low)
                conv_set['mid_mid'] = round(conv_nums * i_mid * w_mid)
                conv_set['mid_high'] = round(conv_nums * i_mid * w_high)

                conv_set['high_low'] = round(conv_nums * i_high * w_low)
                conv_set['high_mid'] = round(conv_nums * i_high * w_mid)
                conv_set['high_high'] = round(conv_nums * i_high * w_high)

                 # 求差值
                conv_sum = 0
                for name in conv_set:
                    # 会偏大
                    conv_sum += conv_set[name] 
                # 实际大小和设置大小的差距 负数
                gap = conv_nums - conv_sum

                # 均摊作差
                count = abs(gap)
                # 单次减多少
                cut = round(count / 6) + 1                

                # 每个乘法操作都有九种可能性
                for name in conv_set:
                    if gap > 0:
                        conv_set[name] = conv_set[name] + gap
                        break
                    else:
                        if (conv_set[name] - cut >= 0) and (count - cut>=0):
                            conv_set[name] -= cut
                            count -= cut

                        elif (conv_set[name] - count >= 0) and (count - cut <= 0):
                            conv_set[name] -= count
                            count = 0
                        
                        if count == 0:
                            break
                
                row_sums = [conv_set[name] for name in conv_set]
                # print(row_sums)

                result_array = self.generate_random_matrix(ow, oh, b, kw, kh, ic,row_sums)
                
                
                compute_sum = 0


                for x in range(batch):

                    # 赋值得到 -> 每一列对应9种计算块分别需要多少个
                    for y,name in zip(range(9),conv_set):
                        conv_set[name] = result_array[y][x]

                    # 找FU的最小切分块: 以此为 cycle 消耗标准
                    # 例如该列的计算需求中有个[8,8]，那么就得按[8,8]来确定cycle

                    min_block = 16

                    for name in conv_set:
                        block_nums = self.get_perf_factor(conv_promote[name]['iprec'], conv_promote[name]['wprec'])
                        if( min_block > block_nums):
                            min_block = block_nums
                    
                    # 以这一列为对象
                    compute_sum += ( ceil_a_by_b(ni, self.N * min_block) + overhead ) * ceil_a_by_b(no, self.M)
                    


                compute_cycles = compute_sum



                # ant
                # compute_cycles = batch * ceil_a_by_b(no, self.M * self.get_perf_factor(wprec)) * \
                #         (ceil_a_by_b(ni, self.N * self.get_perf_factor(iprec)))

            else:
                conv_nums = ow * oh * kw * kh

                conv_set = {}


                conv_set['low_low'] = round(conv_nums * i_low * w_low)
                conv_set['low_mid'] = round(conv_nums * i_low * w_mid)
                conv_set['low_high'] = round(conv_nums * i_low * w_high)

                conv_set['mid_low'] = round(conv_nums * i_mid * w_low)
                conv_set['mid_mid'] = round(conv_nums * i_mid * w_mid)
                conv_set['mid_high'] = round(conv_nums * i_mid * w_high)

                conv_set['high_low'] = round(conv_nums * i_high * w_low)
                conv_set['high_mid'] = round(conv_nums * i_high * w_mid)
                conv_set['high_high'] = round(conv_nums * i_high * w_high)

                # 求差值
                conv_sum = 0
                for name in conv_set:
                    # 会偏大
                    conv_sum += conv_set[name] 
                # 实际大小和设置大小的差距 负数
                gap = conv_nums - conv_sum

                # 均摊作差
                count = abs(gap)
                # 单次减多少
                cut = round(count / 6) + 1                

                # 每个乘法操作都有九种可能性
                for name in conv_set:
                    if gap > 0:
                        conv_set[name] = conv_set[name] + gap
                        break
                    else:
                        if (conv_set[name] - cut >= 0) and (count - cut>=0):
                            conv_set[name] -= cut
                            count -= cut

                        elif (conv_set[name] - count >= 0) and (count - cut <= 0):
                            conv_set[name] -= count
                            count = 0
                        
                        if count == 0:
                            break

                
                # get compute cycle
                sum = 0

                for name in conv_set:
                    sum += conv_set[name] * (
                        ceil_a_by_b(ic, 
                        self.N * self.get_perf_factor(conv_promote[name]['iprec'], conv_promote[name]['wprec']))
                        + overhead
                    )


                compute_cycles = (
                    b
                    * ceil_a_by_b(oc, self.M)
                    * sum
                )
 
        return compute_cycles

    def get_area(self):
        return None
