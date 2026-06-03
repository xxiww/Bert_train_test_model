import pandas as pd

df_train = pd.read_csv('./data/train.csv')
df_dev = pd.read_csv('./data/dev.csv')
df_test = pd.read_csv('./data/test.csv')

print("训练集 label 分布:")
print(df_train['label'].value_counts())
print(f"  0: {len(df_train[df_train['label']==0])}")
print(f"  1: {len(df_train[df_train['label']==1])}")

print("\n验证集 label 分布:")
print(df_dev['label'].value_counts())

print("\n测试集 label 分布:")
print(df_test['label'].value_counts())

print("\n训练集样本示例 (label=0):")
print(df_train[df_train['label']==0]['review'].iloc[1][:100])

print("\n训练集样本示例 (label=1):")
print(df_train[df_train['label']==1]['review'].iloc[1][:100])