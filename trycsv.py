

'''
import csv
import random

# 生成三个数据（a、b、c）
a = [random.uniform(0, 1) for i in range(10)]
b = [random.uniform(0, 1) for i in range(10)]
c = [random.uniform(0, 1) for i in range(10)]

# 将 a、b、c 三个列表的每一个元素作为一个三元组依次写入CSV文件中
# zip() 方法用于将多个可迭代类型结合起来。例如：zip([1, 2, 3], [4, 5, 6]) 返回 [(1, 4), (2, 5), (3, 6)]
with open("data.csv", mode="w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["a", "b", "c"])
    for val in zip(a, b, c):
        writer.writerow(val)

'''        
import csv
import numpy as np

def r2_score(y_true, y_pred):
    """ 
    输入:
        y_true: 数组，包含 n 个目标变量的实际值
        y_pred: 数组，包含 n 个目标变量的预测值
    输出:
        R2：模型的R2值
    """
    y_mean = np.mean(y_true)
    RSS = sum((y_true - y_pred) ** 2)
    TSS = sum((y_true - y_mean) ** 2)
    R2 = 1 - (RSS / TSS)
    return R2

# 获取输入数据和输出数据数组
x_data = []
y_data = []

with open('./0505r2data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    # 跳过csv文件的表头行
    next(reader)

    # 读取每一列的数据并存储到新的列表中
    for row in reader:
        x_data.append(float(row[2]))
        y_data.append(float(row[3]))

# 将列表转换为numpy array对象（方便数学运算）
x_data = np.array(x_data)
y_data = np.array(y_data)

# 使用numpy polyfit计算一次线性回归
p = np.polyfit(x_data, y_data, deg=1)

# 根据线性回归系数 (slope and intercept) 计算预测值
y_pred = p[0]*x_data + p[1]

# 计算R2并打印结果
print("R2 Score:", r2_score(y_data, y_pred))