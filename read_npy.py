# # import numpy as np
# # arr = np.load("/home/joe/AILab/CODES/main_oct_split_1/tsne.npy")
# # print(len(arr))
# # print(len(arr[0]))
# # print(arr[0])
# import csv
# import numpy as np
#
# # 示例字典数据
# data = {
#     'ID': [1, 2, 3],
#     'Matrix': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 随机生成一个3x512的矩阵
# }
# # import numpy as np
# #
# # # 假设 matrix 是你的 numpy 数组
# # matrix =
#
# # 将每个元素转换为字符串，然后使用 '\n'.join() 将每一行连接起来
# # str_matrix = '\n'.join([' '.join(map(str, row)) for row in matrix])
# # print(str_matrix)
# # 将矩阵转换为字符串格式
# def matrix_to_string(matrix):
#     return '\n'.join([' '.join([str(x) for x in row]) for row in matrix])
#
# # 将数据写入CSV文件
# with open('data.csv', 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=data.keys())
#     writer.writeheader()
#     for row in data['Matrix']:
#         writer.writerow({'ID': data['ID'][0], 'Matrix': matrix_to_string(row)})
#         data['ID'] = data['ID'][1:]  # 更新ID列表
import numpy as np
import pandas as pd
# 创建一个76 x 512的数组
array = np.load("/home/joe/AILab/CODES/main_oct_split_1/tsne.npy")
# 将数组转换为DataFrame格式
df = pd.DataFrame(array)
# 将DataFrame写入CSV文件
df.to_csv('array.csv', index=False, header=False)