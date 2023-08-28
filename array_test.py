import numpy as np

def generate_random_matrix(OW, OH, B, KW, KH, IC, row_sums):
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

                elif (final_matrix[y][x] - cut >= 0) and (count - cut <= 0):
                    final_matrix[y][x] -= count
                    count = 0
                
                if count == 0:
                    break


        # final_matrix.sum(axis=0)


    # import ipdb;ipdb.set_trace()

    return final_matrix

# 参数设置
OW = 2
OH = 2
B = 4

KW = 3
KH = 3
IC = 5




# 每一行的总和
row_sums = [12, 8, 10, 6, 10, 14, 10, 10, 10]
for x in range(len(row_sums)):
    row_sums[x] *= 8



# 生成随机数矩阵
random_matrix = generate_random_matrix(OW, OH, B, KW, KH, IC, row_sums)

# 打印结果
print(random_matrix)
