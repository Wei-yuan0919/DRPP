'''
@Project : DeepLearning 
@File    : data_util.py
@IDE     : PyCharm 
@Author  : Kyson. Li
@Date    : 2023/8/25 9:56 
'''
import os

data_dir=r'D:\code\algo_demo\DeepLearning\OCT_model\data_YF'

i=0
for sub_dir in os.listdir(data_dir):
    subdir_path = os.path.join(data_dir, sub_dir)
    #判断字符串按_分割后的第二个元素的第二个字母是否为1
    if sub_dir.split('_')[1][1] == '1':
        i+=1
        #输出正在处理第i个文件夹
        new_sub_dir=sub_dir.replace(sub_dir.split('_')[1],'A_'+sub_dir.split('_')[1])
    if sub_dir.split('_')[1][1] == '2':
        i += 1
        # 输出正在处理第i个文件夹
        new_sub_dir = sub_dir.replace(sub_dir.split('_')[1], 'B_' + sub_dir.split('_')[1])

    if sub_dir.split('_')[1][1] == '3':
        i += 1
        # 输出正在处理第i个文件夹
        new_sub_dir = sub_dir.replace(sub_dir.split('_')[1], 'C_' + sub_dir.split('_')[1])

    print(new_sub_dir)
    new_path=os.path.join(data_dir,new_sub_dir)
    os.rename(subdir_path,new_path)
'''
subdir_path = os.path.join(data_dir, sub_dir)
if os.path.isdir(subdir_path):
    print(subdir_path)
'''