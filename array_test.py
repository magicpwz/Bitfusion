import numpy as np

def generate_matrix_with_prob(OW, OH, B, KW, KH, IC, prob):
    col_sum = KW * KH * IC
    num_elements = col_sum * OW * OH * B
    matrix = np.random.choice(range(1, 10), size=(col_sum, OW * OH * B), p=prob)
    return matrix

# 参数设置
# OW = 4
# OH = 3
# B = 2

OW = 1
OH = 2
B = 1

KW = 2
KH = 2
IC = 1

# 自定义的概率分布
prob = [0.1, 0.1, 0.2, 0.2, 0.05, 0.05, 0.1, 0.1, 0.1]

# 生成矩阵
result_matrix = generate_matrix_with_prob(OW, OH, B, KW, KH, IC, prob)

# 打印结果
print(result_matrix)

conv_promote = {}

# conv_promote[] = {"iprec":2,"wprec":2}
# conv_promote[] = {"iprec":2,"wprec":4}
# conv_promote[] = {"iprec":2,"wprec":8}

# conv_promote[] = {"iprec":4,"wprec":2}
# conv_promote[] = {"iprec":4,"wprec":4}
# conv_promote[] = {"iprec":4,"wprec":8}

# conv_promote[] = {"iprec":8,"wprec":2}
# conv_promote[] = {"iprec":8,"wprec":4}
# conv_promote[] = {"iprec":8,"wprec":8}

a = ["low_low","low_mid","low_high","mid_low","mid_mid","mid_high","high_low","high_mid","high_high"]



import ipdb;ipdb.set_trace()