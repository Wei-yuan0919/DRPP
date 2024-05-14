import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sns.set()
# 读取CSV文件
# PLR
df = pd.read_csv('plr.csv')
# sii
# df = pd.read_csv('sii.csv')
# NLR
# df = pd.read_csv('nlr.csv')

# 提取label和pred列的值
labels = df.iloc[:, 1].tolist()
preds = df.iloc[:, 2].tolist()
probs = df.iloc[:, 3].tolist()
# 计算ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)


# 计算准确率
accuracy = accuracy_score(labels, preds)

# 计算精确率
precision = precision_score(labels, preds, average='binary')

# 计算召回率
recall = recall_score(labels, preds, average='binary')

# 计算F1得分
f1 = f1_score(labels, preds, average='binary')

# auc_roc = roc_auc_score(labels,preds)



print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\n---micro---")

# 计算准确率
accuracy = accuracy_score(labels, preds)

# 计算精确率
precision = precision_score(labels, preds, average='micro')

# 计算召回率
recall = recall_score(labels, preds, average='micro')

# 计算F1得分
f1 = f1_score(labels, preds, average='micro')

# auc_roc = roc_auc_score(labels,preds)



print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\n---macro---")


# 计算准确率
accuracy = accuracy_score(labels, preds)

# 计算精确率
precision = precision_score(labels, preds, average='macro')

# 计算召回率
recall = recall_score(labels, preds, average='macro')

# 计算F1得分
f1 = f1_score(labels, preds, average='macro')

# auc_roc = roc_auc_score(labels,preds)



print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\n---weighted---")

# 计算准确率
accuracy = accuracy_score(labels, preds)

# 计算精确率
precision = precision_score(labels, preds, average='weighted')

# 计算召回率
recall = recall_score(labels, preds, average='weighted')

# 计算F1得分
f1 = f1_score(labels, preds, average='weighted')

# auc_roc = roc_auc_score(labels,preds)



print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# print("auc-roc",auc_roc)

C2= confusion_matrix(labels, preds)
print(C2)
plt.figure(figsize=(10,7))
sns.heatmap(C2, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
