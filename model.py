import pandas as pd
import numpy as np
import os
import sys
import joblib
import chardet  # 用于自动检测文件编码
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder  # 替换get_dummies，解决编码乱码和特征匹配问题
import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告

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

# ===================== 工具函数（解决乱码核心） =====================
def detect_file_encoding(file_path):
    """自动检测文件编码，避免读取乱码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10240)  # 读取前10KB检测编码
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    # 兼容常见编码别名
    if encoding == 'GB2312':
        encoding = 'GBK'
    elif encoding is None:
        encoding = 'utf-8-sig'
    return encoding

def safe_save_model(obj, path, compress_level=5):
    """安全保存模型，避免字符编码乱码，同时最大化压缩"""
    # 确保路径为字符串且编码正确
    if isinstance(path, str):
        path = path.encode('utf-8').decode('utf-8')
    # 高压缩比保存（1-9，5是平衡值，9压缩最大但保存稍慢）
    joblib.dump(obj, path, compress=compress_level)

# ===================== 1. 加载并校验数据（彻底解决乱码） =====================
try:
    # 自动检测编码，彻底解决CSV读取乱码
    file_encoding = detect_file_encoding(file_path)
    df = pd.read_csv(file_path, encoding=file_encoding)
    print(f"✅ 自动检测文件编码：{file_encoding}")
    
    # 清理列名（统一格式，避免符号/编码问题）
    df.columns = (
        df.columns
        .str.strip()
        .str.replace('（小时）', '', regex=False)
        .str.replace('（', '(', regex=False)
        .str.replace('）', ')', regex=False)
        .str.encode('utf-8').str.decode('utf-8')  # 确保列名编码正确
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
    sys.exit(1)  # 替代exit()，更安全的退出方式

# ===================== 2. 训练数值预测模型（进一步轻量化） =====================
try:
    X_reg = df[['每周学习时长', '上课出勤率', '期中考试分数', '作业完成率']]
    y_reg = df['期末考试分数']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    # 线性回归模型本身体积极小，无需额外轻量化
    reg_model = LinearRegression()
    reg_model.fit(X_train_reg, y_train_reg)
    # 输出模型评估指标（新增）
    reg_score = reg_model.score(X_test_reg, y_test_reg)
    print(f"✅ 数值预测模型训练成功，测试集R²得分：{reg_score:.4f}")
except Exception as e:
    print(f"❌ 数值模型训练失败：{str(e)}")
    sys.exit(1)

# ===================== 3. 训练分类预测模型（极致轻量化+解决编码乱码） =====================
try:
    # 新增成绩等级标签（及格/不及格）
    df['成绩等级'] = df['期末考试分数'].apply(lambda x: 1 if x >= 60 else 0)
    
    # 校验分类标签是否有两种（避免全及格/全不及格）
    if df['成绩等级'].nunique() < 2:
        raise ValueError("成绩等级仅有一种（全及格/全不及格），无法训练分类模型")
    
    # 分离类别特征和数值特征（解决get_dummies编码乱码问题）
    cat_features = ['性别', '专业']
    num_features = ['每周学习时长', '上课出勤率', '期中考试分数', '作业完成率']
    
    # 使用OneHotEncoder替代get_dummies，解决特征编码乱码和匹配问题
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = encoder.fit_transform(df[cat_features])
    cat_feature_names = encoder.get_feature_names_out(cat_features)
    
    # 拼接特征
    X_clf = np.hstack([df[num_features].values, cat_encoded])
    X_clf_df = pd.DataFrame(X_clf, columns=list(num_features) + list(cat_feature_names))
    y_clf = df['成绩等级']
    
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # 极致轻量化随机森林参数（进一步减小体积）
    clf_model = RandomForestClassifier(
        n_estimators=30,        # 树数量从50减到30，大幅减小体积
        max_depth=8,            # 树深度从10减到8
        min_samples_split=8,    # 分裂最小样本数增加
        min_samples_leaf=3,     # 叶节点最小样本数增加
        max_features='sqrt',    # 限制每次分裂使用的特征数
        n_jobs=1,               # 单线程训练，减小模型体积（多线程会增加序列化体积）
        random_state=42,
        verbose=0
    )
    clf_model.fit(X_train_clf, y_train_clf)
    
    # 输出模型评估指标（新增）
    clf_score = clf_model.score(X_test_clf, y_test_clf)
    print(f"✅ 分类预测模型训练成功，测试集准确率：{clf_score:.4f}")
except Exception as e:
    print(f"❌ 分类模型训练失败：{str(e)}")
    sys.exit(1)

# ===================== 4. 压缩保存模型（解决乱码+最小体积） =====================
try:
    # 保存数值预测模型（高压缩比）
    reg_model_path = os.path.join(model_dir, "linear_regression_model.pkl")
    safe_save_model(reg_model, reg_model_path, compress_level=9)  # 9为最高压缩比
    print(f"✅ 数值模型已保存至：{reg_model_path}")
    if os.path.exists(reg_model_path):
        reg_size = round(os.path.getsize(reg_model_path)/1024, 2)
        print(f"   ✔️ 数值模型大小：{reg_size} KB")
    
    # 保存分类预测模型（高压缩比）
    clf_model_path = os.path.join(model_dir, "random_forest_clf.pkl")
    safe_save_model(clf_model, clf_model_path, compress_level=9)
    print(f"✅ 分类模型已保存至：{clf_model_path}")
    if os.path.exists(clf_model_path):
        clf_size = round(os.path.getsize(clf_model_path)/1024, 2)
        print(f"   ✔️ 分类模型大小：{clf_size} KB")
    
    # 保存编码器和特征列名（解决预测时编码乱码）
    encoder_path = os.path.join(model_dir, "onehot_encoder.pkl")
    safe_save_model(encoder, encoder_path, compress_level=9)
    feature_cols_path = os.path.join(model_dir, "clf_feature_cols.pkl")
    safe_save_model(X_clf_df.columns, feature_cols_path, compress_level=9)
    print(f"✅ 编码器/特征列名已保存，解决预测时编码问题")

except Exception as e:
    print(f"❌ 模型保存失败：{str(e)}")
    sys.exit(1)

# ===================== 验证文件生成 =====================
print("\n🎉 所有模型保存完成！")
print("\n📋 模型目录下的.pkl文件列表：")
total_size = 0
for file in os.listdir(model_dir):
    if file.endswith('.pkl'):
        file_path_full = os.path.join(model_dir, file)
        file_size = round(os.path.getsize(file_path_full)/1024, 2)
        total_size += file_size
        print(f"   - {file} | 大小：{file_size} KB")
print(f"\n📊 所有模型文件总大小：{total_size:.2f} KB")
