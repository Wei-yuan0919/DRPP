'''
@Project ：algo_demo 
@File    ：log_utils.py
@IDE     ：PyCharm 
@Author  ：Kyson. Li
@Date    ：2023/8/9 15:07 
'''

#每个epoch结束后保存模型，同时保存模型的参数，以及模型的训练结果
#分别为训练集和测试集的
'''

1. sample_result.csv  训练集和测试集每个样本的预测结果及特征，以及底层对应该组结果的评价指标
   每个epoch记录下保存的csv文件
    samples['id']=id
    samples['img_path']=img_path
    samples['ground_truth']=ground_truth
    samples['pred']=pred
    samples['pred_name']=predict_name
    samples['prob']=prob
    samples['feature']=feature
    
2. overall_training.csv 训练集和测试集的评价指标随epoch训练的变化情况
   
   
   二分类任务的评价指标
    results['epoch']=epoch
    results['accuracy']=accuracy
    results['npv']=npv
    results['ppv']=ppv
    results['specificity'] = specificity
    results['sensitivity'] = sensitivity
    results['precision']=precision
    results['recall']=recall
    results['f1']=f1
    results['tn']=tn
    results['fp']=fp
    results['fn']=fn
    results['tp']=tp
    results['auc']=auc
    
    多分类任务的评价指标
    results['epoch']=epoch
    results['accuracy']=accuracy
    results['npv']=npv
    results['ppv']=ppv
    results['specificity'] = specificity
    results['sensitivity'] = sensitivity
    results['precision']=precision
    results['recall']=recall
    results['f1']=f1
    results['tn']=tn
    results['fp']=fp
    results['fn']=fn
    results['tp']=tp
    results['auc']=auc
    results['img_numbers']=img_numbers
    
    多分类任务的评价指标
'''


import csv

def save_dict_to_csv(samples,current_epoch,save_dir):
    # 获取所有键值
    filename=save_dir+'/sample_result_epoch_'+str(current_epoch)+'.csv'
    keys = list(samples.keys())

    # 获取最长的列表长度
    max_length = max(len(samples[key]) for key in keys)

    # 添加ID列

    keys.insert(0, 'ID')
    titles = []
    titles.append('ID')
    # 创建包含数据的列表
    data = []

    for key in keys[1:]:
        if key == 'prob':
            prob_len=len(samples[key][0])
            for i in range(prob_len):
                titles.append('prob_'+str(i))
        elif key == 'features':
            feature_len = len(samples[key][0])
            for i in range(feature_len):
                titles.append('feature_' + str(i))
        else:
            titles.append(key)


    # 填充数据列表
    for i in range(max_length):
        row = [i + 1]  # ID列从1开始
        for key in keys[1:]:

            if key == 'prob' and i < len(samples[key]):
                row.extend(samples[key][i])

            elif key=='features' and i < len(samples[key]):
                row.extend(samples[key][i])
                # if row==max_length:
                #     titles.append('features')
            elif i < len(samples[key]):
                row.append(samples[key][i])
            else:
                row.append('')
        data.append(row)

    # 将数据写入CSV文件
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(titles)  # 写入列名
        writer.writerows(data)

    print('Saved samples to {}'.format(filename))


import csv

import os


min_sii= 62.9762533
max_sii= 2088.98


def save_label_pred_to_csv2(label, patient_id, pred, probs, current_epoch,acc, mae, save_dir):
    # 构建CSV文件路径
    # pred=[i[0] for i in pred]
    siis=[]
    p_siis=[]
    for i in range(len(pred)):
        lb=label[i]
        prd=pred[i]
        prob = probs[i]
        sii=lb*(max_sii-min_sii)+min_sii
        p_sii=prd*(max_sii-min_sii)+min_sii
        siis.append(sii)
        p_siis.append(p_sii)


    filename = f"sample_result_epoch_{current_epoch}_acc_{acc}_mae_{mae}_testing.csv"
    filepath = os.path.join(save_dir, filename)

    # 创建一个包含label和pred的列表，每个元素为一行数据
    data = list(zip(patient_id,label, pred,probs,siis,p_siis ))

    print('patient_id',patient_id)
    print('label',label)
    print('pred',pred)
    print("probs",probs)
    print('siis',siis)
    print('p_siis',p_siis)



    # 打开CSV文件并写入数据
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入列名
        writer.writerow(["patient_id","label", "pred", "probs", "sii","p_sii"])

        # 写入行数据
        writer.writerows(data)

    print(f"Label and pred saved to {filepath}")







def save_label_pred_to_csv1(label, patient_id,pred, current_epoch,acc, mae, save_dir):
    # 构建CSV文件路径
    # pred=[i[0] for i in pred]
    siis=[]
    p_siis=[]
    for i in range(len(pred)):
        lb=label[i]
        prd=pred[i]
        sii=lb*(max_sii-min_sii)+min_sii
        p_sii=prd*(max_sii-min_sii)+min_sii
        siis.append(sii)
        p_siis.append(p_sii)


    filename = f"sample_result_epoch_{current_epoch}_acc_{acc}_mae_{mae}_testing.csv"
    filepath = os.path.join(save_dir, filename)

    # 创建一个包含label和pred的列表，每个元素为一行数据
    data = list(zip(patient_id,label, pred,siis,p_siis ))

    print('patient_id',patient_id)
    print('label',label)
    print('pred',pred)
    print('siis',siis)
    print('p_siis',p_siis)



    # 打开CSV文件并写入数据
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入列名
        writer.writerow(["patient_id","label", "pred","sii","p_sii"])

        # 写入行数据
        writer.writerows(data)

    print(f"Label and pred saved to {filepath}")



def save_label_pred_to_csv(label, patient_id,pred, current_epoch, mae, mse, save_dir):
    # 构建CSV文件路径
    # print()
    print("pred",pred)
    pred=[i[0] for i in pred]
    print("pred",pred)
    siis=[]
    p_siis=[]
    for i in range(len(pred)):
        lb=label[i]
        prd=pred[i]
        sii=lb*(max_sii-min_sii)+min_sii
        p_sii=prd*(max_sii-min_sii)+min_sii
        siis.append(sii)
        p_siis.append(p_sii)


    filename = f"sample_result_epoch_{current_epoch}_mae_{mae}_mse_{mse}_testing.csv"
    filepath = os.path.join(save_dir, filename)

    # 创建一个包含label和pred的列表，每个元素为一行数据
    data = list(zip(patient_id,label, pred,siis,p_siis ))

    print('patient_id',patient_id)
    print('label',label)
    print('pred',pred)
    print('siis',siis)
    print('p_siis',p_siis)



    # 打开CSV文件并写入数据
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入列名
        writer.writerow(["patient_id","label", "pred","sii","p_sii"])

        # 写入行数据
        writer.writerows(data)

    print(f"Label and pred saved to {filepath}")


def save_dict_to_csv_dict(sample_results,mae,mse,training,current_epoch,save_dir):

    if training:
        filename=save_dir+'/sample_result_epoch_'+str(current_epoch)+'_training.csv'
    else:
        filename = save_dir + '/sample_result_epoch_' + str(current_epoch) + '_mae_'+mae+'_mse_'+mse+'_testing.csv'
    titles = []
    titles.append('ID')
    # 创建包含数据的列表
    data = []
    set_mark=[]
    # 获取所有键值
    set_keys = list(sample_results.keys())
    for set_key in set_keys:
        samples=sample_results[set_key]
        set_mark.append([set_key])
        # 获取最长的列表长度
        max_length = max(len(samples[key]) for key in samples.keys())
        keys = list(samples.keys())
        keys.insert(0, 'ID')
        if training:
            if set_key=='training_set':
                for key in keys[1:]:
                    if key == 'prob':
                        prob_len = len(samples[key][0])
                        for i in range(prob_len):
                            titles.append('prob_' + str(i))
                    elif key == 'features':
                        feature_len = len(samples[key][0])
                        for i in range(feature_len):
                            titles.append('feature_' + str(i))
                    else:
                        titles.append(key)
        elif set_key=='testing_set':
                for key in keys[1:]:
                    if key == 'prob':
                        prob_len = len(samples[key][0])
                        for i in range(prob_len):
                            titles.append('prob_' + str(i))
                    elif key == 'features':
                        feature_len = len(samples[key][0])
                        for i in range(feature_len):
                            titles.append('feature_' + str(i))
                    else:
                        titles.append(key)

        if set_key == 'training_set':
            data.append([set_key])
            data.append(titles)
        if set_key=='testing_set':
            data.append([set_key])
            data.append(titles)
            # 填充数据列表
        for i in range(max_length):
            row = [i + 1]  # ID列从1开始
            for key in keys[1:]:

                if key == 'prob' and i < len(samples[key]):
                    row.extend(samples[key][i])

                elif key == 'features' and i < len(samples[key]):
                    row.extend(samples[key][i])
                    # if row==max_length:
                    #     titles.append('features')
                elif i < len(samples[key]):
                    row.append(samples[key][i])
                else:
                    row.append('')
            data.append(row)

    #data.append(row)
    # 将数据写入CSV文件
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        #writer.writerow(set_mark)   #写入set_name
        #writer.writerow(titles)  # 写入列名
        writer.writerows(data)

    print('Saved samples to {}'.format(filename))



if __name__ == '__main__':
    sample_results={"training_set":[],"test_set":[]}
    save_dict_to_csv_dict(sample_results,1,'./')

#save_dict_to_csv(samples, 'output.csv')