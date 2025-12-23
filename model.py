import pandas as pd
import numpy as np
import os
import joblib  # 替换pickle，更适合sklearn模型且压缩效果更好
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# ===================== 基础配置与目录检查 =====================
# 定义路径（r前缀避免转义）
file_path = r"D:\streamlit_env\student_data_adjusted_rounded.csv"
model_dir = r"D:\streamlit_env"

# 检查并创建模型保存目录
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"📁 目录不存在，已创建：{model_dir}")
else:
    print(f"📁 目录已存在：{model_dir}")

# ===================== 1. 加载并校验数据 =====================
try:
    # 兼容不同编码的CSV文件
    try:
        df = pd.read_csv(file_path, encoding='gbk')
    except:
        df = pd.read_csv(file_path, encoding='utf-8')
    
    # 清理列名
    df.columns = (
        df.columns
        .str.strip()
        .str.replace('（小时）', '', regex=False)
        .str.replace('（', '(', regex=False)
        .str.replace('）', ')', regex=False)
    )
    
    # 校验必要列是否存在
    required_cols = ['学号', '性别', '专业', '每周学习时长', '上课出勤率', '期中考试分数', '作业完成率', '期末考试分数']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据文件缺少必要列：{missing_cols}")
    
    # 清洗数据
    df = df[required_cols].dropna()
    numeric_cols = ['每周学习时长', '上课出勤率', '期中考试分数', '作业完成率', '期末考试分数']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    # 校验数据量（至少10条才能训练）
    if len(df) < 10:
        raise ValueError(f"有效数据量过少，仅{len(df)}条，无法训练模型")
    
    print(f"✅ 数据加载成功，有效数据量：{len(df)}条")
except Exception as e:
    print(f"❌ 数据加载/处理失败：{str(e)}")
    exit()

# ===================== 2. 训练数值预测模型（线性回归） =====================
try:
    X_reg = df[['每周学习时长', '上课出勤率', '期中考试分数', '作业完成率']]
    y_reg = df['期末考试分数']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_model = LinearRegression()
    reg_model.fit(X_train_reg, y_train_reg)
    print("✅ 数值预测模型训练成功")
except Exception as e:
    print(f"❌ 数值模型训练失败：{str(e)}")
    exit()

# ===================== 3. 训练分类预测模型（轻量化随机森林） =====================
try:
    # 新增成绩等级标签（及格/不及格）
    df['成绩等级'] = df['期末考试分数'].apply(lambda x: 1 if x >= 60 else 0)
    
    # 校验分类标签是否有两种（避免全及格/全不及格）
    if df['成绩等级'].nunique() < 2:
        raise ValueError("成绩等级仅有一种（全及格/全不及格），无法训练分类模型")
    
    # 特征编码
    X_clf = pd.get_dummies(df[['性别', '专业', '每周学习时长', '上课出勤率', '期中考试分数', '作业完成率']])
    y_clf = df['成绩等级']
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # 轻量化随机森林参数（核心：减少树数量+限制树深度，大幅减小体积）
    clf_model = RandomForestClassifier(
        n_estimators=50,       # 树的数量从100减到50（体积减半）
        max_depth=10,          # 限制树的深度（避免过拟合+减小体积）
        min_samples_split=5,   # 增加分裂最小样本数（简化树结构）
        min_samples_leaf=2,    # 增加叶节点最小样本数
        random_state=42
    )
    clf_model.fit(X_train_clf, y_train_clf)
    print("✅ 分类预测模型（轻量化）训练成功")
except Exception as e:
    print(f"❌ 分类模型训练失败：{str(e)}")
    exit()

# ===================== 4. 压缩保存模型（核心：减小文件体积） =====================
try:
    # 保存数值预测模型（joblib压缩，compress=3平衡压缩率和速度）
    reg_model_path = os.path.join(model_dir, "linear_regression_model.pkl")
    joblib.dump(reg_model, reg_model_path, compress=3)
    print(f"✅ 数值模型已保存至：{reg_model_path}")
    if os.path.exists(reg_model_path):
        print(f"   ✔️ 数值模型大小：{round(os.path.getsize(reg_model_path)/1024, 2)} KB")
    
    # 保存分类预测模型（高压缩比）
    clf_model_path = os.path.join(model_dir, "random_forest_clf.pkl")
    joblib.dump(clf_model, clf_model_path, compress=3)  # compress=1~9，3是最优平衡
    print(f"✅ 分类模型已保存至：{clf_model_path}")
    if os.path.exists(clf_model_path):
        print(f"   ✔️ 分类模型大小：{round(os.path.getsize(clf_model_path)/1024, 2)} KB")
    
    # 保存分类模型的特征列名（预测时必须）
    feature_cols_path = os.path.join(model_dir, "clf_feature_cols.pkl")
    joblib.dump(X_clf.columns, feature_cols_path, compress=3)
    print(f"✅ 特征列名已保存至：{feature_cols_path}")
    if os.path.exists(feature_cols_path):
        print(f"   ✔️ 特征列名文件大小：{round(os.path.getsize(feature_cols_path)/1024, 2)} KB")

except Exception as e:
    print(f"❌ 模型保存失败：{str(e)}")
    exit()

# ===================== 验证文件生成 =====================
print("\n🎉 所有模型保存完成！")
print("\n📋 模型目录下的.pkl文件列表：")
for file in os.listdir(model_dir):
    if file.endswith('.pkl'):
        file_size = round(os.path.getsize(os.path.join(model_dir, file))/1024, 2)
        print(f"   - {file} | 大小：{file_size} KB")
