#!/usr/bin/env python
# Author  : KerryChen
# File    : plot_lines.py
# Time    : 2025/1/20 20:53

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

# 初始化一个列表，用于存储每个文件中 BLEU 值最大的行
best_rows = []

file_dir = '/home/qfchen/CPPCGM/CPPGenerator/results'
# 遍历文件名，从 training_results_1.csv 到 training_results_9.csv
for i in range(1, 10):
    file_name = '/home/qfchen/CPPCGM/CPPGenerator/results/' + f'training_results_{i}.csv'
    # 读取 CSV 文件
    df = pd.read_csv(file_name)

    # 找到 BLEU 列最大值的行
    best_row = df.loc[df['BLEU'].idxmax()]
    best_rows.append(best_row)

# 将 best_rows 转换为 DataFrame 以便于绘图
best_df = pd.DataFrame(best_rows)

# 提取需要绘制的数据
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 横坐标
y_bleu = best_df['BLEU'].values  # BLEU 值
y_rouge1 = best_df['ROUGE-1'].values  # ROUGE-1 值
y_rouge2 = best_df['ROUGE-2'].values  # ROUGE-2 值
y_rougel = best_df['ROUGE-L'].values  # ROUGE-L 值

# 绘制折线图
plt.figure(figsize=(10, 6))

# 绘制 BLEU 曲线
plt.plot(x, y_bleu, label='BLEU', marker='o')
# 绘制 ROUGE-1 曲线
plt.plot(x, y_rouge1, label='ROUGE-1', marker='o')
# 绘制 ROUGE-2 曲线
plt.plot(x, y_rouge2, label='ROUGE-2', marker='o')
# 绘制 ROUGE-L 曲线
plt.plot(x, y_rougel, label='ROUGE-L', marker='o')

# 添加图例
plt.legend(fontsize=12)

# 添加标题和坐标轴标签
plt.xlabel('Perturbation Rate', fontsize=15, weight='bold')
plt.ylabel('Scores', fontsize=15, weight='bold')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
# 显示网格
# plt.grid(True)

# save figures
plt.savefig('/home/qfchen/CPPCGM/CPPGenerator/results/plot_lines.png', bbox_inches='tight', dpi=600)
plt.savefig('/home/qfchen/CPPCGM/CPPGenerator/results/plot_lines.pdf', bbox_inches='tight')


# 显示图表
plt.show()
