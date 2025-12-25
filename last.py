import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import chardet  # 新增：自动检测文件编码解决乱码

# -------------------------- 基础配置 --------------------------
st.set_page_config(
    page_title="学生成绩分析与预测系统",
    page_icon="📊",
    layout="wide"
)

# 中文显示配置（兼容更多系统）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# -------------------------- 工具函数（核心修复） --------------------------
def detect_file_encoding(file_path):
    """自动检测文件编码，彻底解决CSV乱码"""
    if not os.path.exists(file_path):
        return 'utf-8-sig'
    with open(file_path, 'rb') as f:
        raw_data = f.read(10240)
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8-sig'
    return 'gbk' if encoding == 'GB2312' else encoding

def safe_load_model(model_path):
    """安全加载模型，避免路径/编码乱码"""
    if not os.path.exists(model_path):
        st.warning(f"模型文件不存在：{model_path}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"加载模型{os.path.basename(model_path)}失败：{str(e)}")
        return None

# -------------------------- 数据与模型加载（修复乱码+特征匹配） --------------------------
@st.cache_data(ttl=3600)  # 延长缓存时间，避免重复加载
def load_data_and_models():
    # 数据加载（修复乱码）
    file_name = "student_data_adjusted_rounded.csv"
    df = None
    if not os.path.exists(file_name):
        st.error(f"数据文件不存在：{os.path.abspath(file_name)}")
        return None, None, None, None, None

    try:
        # 自动检测编码读取CSV
        file_encoding = detect_file_encoding(file_name)
        df = pd.read_csv(file_name, encoding=file_encoding)
        
        # 列名清理（统一编码+格式）
        df.columns = (
            df.columns
            .str.strip()
            .str.replace('（小时）', '', regex=False)
            .str.replace('（', '(', regex=False)
            .str.replace('）', ')', regex=False)
            .str.encode('utf-8').str.decode('utf-8')  # 确保中文列名编码正确
        )
        
        # 数据清洗
        required_cols = ['学号', '性别', '专业', '每周学习时长', '上课出勤率', '期中考试分数', '作业完成率', '期末考试分数']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"数据文件缺少必要列：{missing_cols}")
            return None, None, None, None, None
        
        df = df[required_cols].dropna()
        numeric_cols = ['每周学习时长', '上课出勤率', '期中考试分数', '作业完成率', '期末考试分数']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
    except Exception as e:
        st.error(f"数据加载失败：{str(e)}")
        return None, None, None, None, None

    # 模型加载（修复路径+编码）
    reg_model = safe_load_model("linear_regression_model.pkl")
    clf_model = safe_load_model("random_forest_clf.pkl")
    clf_feature_cols = safe_load_model("clf_feature_cols.pkl")
    encoder = safe_load_model("onehot_encoder.pkl")  # 新增：加载编码器

    return df, reg_model, clf_model, clf_feature_cols, encoder

# 初始化数据和模型（新增编码器）
student_df, reg_model, clf_model, clf_feature_cols, encoder = load_data_and_models()

# -------------------------- 辅助函数（修复特征匹配bug） --------------------------
def build_clf_input(input_data, encoder, num_features, cat_features):
    """重构分类模型输入（用OneHotEncoder替代get_dummies，解决特征不匹配）"""
    # 构造基础DataFrame
    raw_df = pd.DataFrame({
        '性别': [input_data['gender']],
        '专业': [input_data['major']],
        '每周学习时长': [input_data['study_hour']],
        '上课出勤率': [input_data['attendance']],
        '期中考试分数': [input_data['mid_score']],
        '作业完成率': [input_data['homework_rate']]
    })
    
    # 分离数值/类别特征
    num_df = raw_df[num_features]
    cat_df = raw_df[cat_features]
    
    # 使用训练好的编码器（避免特征不匹配）
    cat_encoded = encoder.transform(cat_df)
    cat_feature_names = encoder.get_feature_names_out(cat_features)
    
    # 拼接特征
    encoded_df = pd.DataFrame(cat_encoded, columns=cat_feature_names)
    final_df = pd.concat([num_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
    # 补全特征列（兼容训练时的列名）
    if clf_feature_cols is not None:
        for col in clf_feature_cols:
            if col not in final_df.columns:
                final_df[col] = 0
        final_df = final_df[clf_feature_cols]
    
    return final_df

# -------------------------- 界面1：项目介绍 --------------------------
def show_project_intro():
    st.title("学生成绩分析与预测系统")
    st.divider()

    col_content, col_img = st.columns([5, 3])
    with col_content:
        st.subheader("📋 项目概述")
        st.write("""
        本系统基于Streamlit开发，整合机器学习模型实现学生期末成绩的精准预测，
        同时提供多维度的专业成绩分析功能，帮助教师/学生掌握学习情况。
        """)

        st.subheader("✨ 核心功能")
        core_functions = [
            "📊 专业维度分析：性别比例、成绩趋势、出勤率统计",
            "🎯 双模型预测：数值分数预测 + 及格状态预测",
            "📈 专项分析：大数据管理专业成绩/学习时长分布"
        ]
        for func in core_functions:
            st.write(func)

        st.subheader("🎯 项目目标")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.write("**精准分析**")
            st.write("- 识别关键学习影响因素")
            st.write("- 量化学习时长/出勤率对成绩的影响")
        with col_t2:
            st.write("**直观展示**")
            st.write("- 可视化成绩分布趋势")
            st.write("- 清晰呈现各专业差异")
        with col_t3:
            st.write("**个性化预测**")
            st.write("- 基于多维度特征预测成绩")
            st.write("- 匹配对应等级表情包")

        st.subheader("🔧 技术栈")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.write("**前端框架**")
            st.write("Streamlit")
        with col_s2:
            st.write("**数据处理**")
            st.write("Pandas / NumPy")
        with col_s3:
            st.write("**可视化**")
            st.write("Matplotlib")
        with col_s4:
            st.write("**机器学习**")
            st.write("Scikit-learn")

    with col_img:
        st.write("### 系统预览")
        # 修复图片缺失问题
        preview_img = "系统预览图.png"
        if os.path.exists(preview_img):
            st.image(preview_img, use_container_width=True)
        else:
            st.info("📌 系统预览图未找到（请放置系统预览图.png到当前目录）")

# -------------------------- 界面2：专业数据分析 --------------------------
def show_major_analysis():
    if student_df is None:
        st.warning("⚠️ 数据加载失败，无法进行分析！")
        return
    
    st.title("专业成绩数据分析")
    st.divider()

    # 1. 性别比例分析
    st.subheader("1. 各专业男女性别比例")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        gender_stats = student_df.groupby('专业')['性别'].value_counts(normalize=True).unstack().fillna(0)
        gender_stats.columns = ['男性比例', '女性比例'] if '男' in gender_stats.columns else gender_stats.columns
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        gender_stats.plot(kind='bar', stacked=True, ax=ax1, color=['#1f77b4', '#ff7f0e'])
        ax1.set_xlabel("专业")
        ax1.set_ylabel("比例")
        ax1.set_title("各专业男女性别分布")
        ax1.legend(loc='upper right')
        ax1.grid(alpha=0.3, axis='y')
        st.pyplot(fig1)
    with col_right:
        st.write("### 性别比例数据")
        st.dataframe(gender_stats.round(4))

    # 2. 学习指标对比
    st.subheader("2. 各专业学习指标对比")
    st.write("（期中/期末成绩 + 每周学习时长）")
    col_chart2, col_table2 = st.columns([3, 1])
    with col_chart2:
        study_stats = student_df.groupby('专业').agg({
            '期中考试分数': 'mean',
            '期末考试分数': 'mean',
            '每周学习时长': 'mean'
        }).round(4)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        study_stats[['期中考试分数', '期末考试分数']].plot(
            kind='line', marker='o', ax=ax2, color=['#1f77b4', '#d62728']
        )
        ax2_right = ax2.twinx()
        study_stats['每周学习时长'].plot(
            kind='line', marker='s', ax=ax2_right, color='#2ca02c', linewidth=2
        )
        
        ax2.set_xlabel("专业")
        ax2.set_ylabel("分数")
        ax2_right.set_ylabel("每周学习时长（小时）")
        ax2.set_title("各专业成绩与学习时长趋势")
        ax2.grid(alpha=0.3)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        st.pyplot(fig2)
    with col_table2:
        st.write("### 学习指标数据")
        st.dataframe(study_stats)

    # 3. 出勤率分析
    st.subheader("3. 各专业出勤率分析")
    col_chart3, col_table3 = st.columns([3, 1])
    with col_chart3:
        attendance_stats = student_df.groupby('专业')['上课出勤率'].mean().round(4).to_frame('平均出勤率')
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        attendance_stats.plot(kind='bar', ax=ax3, color='#2ca02c')
        ax3.set_xlabel("专业")
        ax3.set_ylabel("平均出勤率")
        ax3.set_title("各专业平均上课出勤率")
        ax3.grid(alpha=0.3, axis='y')
        st.pyplot(fig3)
    with col_table3:
        st.write("### 出勤率数据")
        st.dataframe(attendance_stats)

    # 4. 大数据管理专项分析
    st.subheader("4. 大数据管理专业专项分析")
    if '大数据管理' in student_df['专业'].unique():
        bd_df = student_df[student_df['专业'] == '大数据管理']
        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("平均出勤率", f"{bd_df['上课出勤率'].mean():.1%}")
        with col2:
            st.metric("平均期末分数", f"{bd_df['期末考试分数'].mean():.1f}")
        with col3:
            st.metric("通过率", f"{(bd_df['期末考试分数'] >= 60).mean():.1%}")
        with col4:
            st.metric("平均学习时长", f"{bd_df['每周学习时长'].mean():.1f}小时")
        
        # 分布图表
        col1, col2 = st.columns(2)
        with col1:
            st.write("期末成绩分布")
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            ax4.hist(bd_df['期末考试分数'], bins=10, color='#1f77b4')
            ax4.set_xlabel("分数")
            ax4.set_ylabel("人数")
            st.pyplot(fig4)
        with col2:
            st.write("每周学习时长分布")
            fig5, ax5 = plt.subplots(figsize=(5, 4))
            ax5.boxplot(bd_df['每周学习时长'], vert=False)
            ax5.set_xlabel("时长（小时）")
            st.pyplot(fig5)
    else:
        st.info("📌 当前数据集无「大数据管理」专业数据")

# -------------------------- 界面3：成绩预测（修复核心bug） --------------------------
def show_score_prediction():
    if reg_model is None or clf_model is None or encoder is None:
        st.warning("⚠️ 模型/编码器加载不完整，无法进行成绩预测！")
        return
    
    st.title("期末成绩预测")
    st.write("📝 输入学生信息，系统将预测期末成绩并匹配对应等级表情包")
    st.divider()

    # 输入表单
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("学号", value="2024001")
        gender = st.selectbox("性别", ["男", "女"], index=0)
        # 兼容专业列表为空的情况
        major_options = student_df['专业'].unique() if student_df is not None else ["大数据管理"]
        major = st.selectbox("专业", major_options, index=0)
        submit_btn = st.button("🚀 预测期末成绩", type="primary")
    with col2:
        study_hour = st.slider("每周学习时长（小时）", 0.0, 50.0, 15.0, 0.1)
        attendance = st.slider("上课出勤率", 0.0, 1.0, 0.9, 0.01)
        mid_score = st.slider("期中考试分数", 0.0, 100.0, 70.0, 0.1)
        homework_rate = st.slider("作业完成率", 0.0, 1.0, 0.95, 0.01)

    # 预测逻辑（修复特征构建）
    if submit_btn:
        try:
            # 1. 数值分数预测
            reg_input = np.array([[study_hour, attendance, mid_score, homework_rate]])
            pred_score = reg_model.predict(reg_input)[0]
            pred_score = np.clip(pred_score, 0, 100)  # 限制分数在0-100之间

            # 2. 分类预测（修复特征匹配）
            input_data = {
                'gender': gender,
                'major': major,
                'study_hour': study_hour,
                'attendance': attendance,
                'mid_score': mid_score,
                'homework_rate': homework_rate
            }
            # 定义特征列表（与训练时一致）
            num_features = ['每周学习时长', '上课出勤率', '期中考试分数', '作业完成率']
            cat_features = ['性别', '专业']
            # 构建分类模型输入
            clf_input = build_clf_input(input_data, encoder, num_features, cat_features)
            pred_clf = clf_model.predict(clf_input)[0]

            # 3. 结果判断
            pred_clf_label = "及格" if pred_score >= 60 else "不及格"
            delta_text = "达标" if pred_score >= 60 else "需提升"

            # 4. 结果展示
            st.success("🎉 成绩预测完成！")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="预测期末分数",
                    value=f"{pred_score:.2f}分",
                    delta=f"与基准分70分偏差 ±{abs(pred_score-70):.1f}分"
                )
            with col2:
                st.metric(
                    label="及格状态",
                    value=pred_clf_label,
                    delta=delta_text
                )

            # 5. 等级匹配（修复图片缺失问题）
            st.subheader("📊 成绩等级匹配")
            level_mapping = {
                "不及格": ("未及格.PNG", 0, 59),
                "及格": ("及格.PNG", 60, 69),
                "良好": ("良好.PNG", 70, 89),
                "优秀": ("优秀.PNG", 90, 100)
            }
            # 判断等级
            level = "不及格"
            for key, (_, min_score, max_score) in level_mapping.items():
                if min_score <= pred_score <= max_score:
                    level = key
                    break
            # 展示图片（兼容缺失）
            img_path = level_mapping[level][0]
            if os.path.exists(img_path):
                st.image(img_path, width=150)
            else:
                st.info(f"📌 等级图片缺失：{img_path}（当前等级：{level}）")

        except Exception as e:
            st.error(f"❌ 预测失败：{str(e)}")
            st.info("提示：请确保训练模型时的特征与预测输入特征一致，或重新训练模型")

# -------------------------- 导航菜单 --------------------------
st.sidebar.title("📚 系统导航")
page = st.sidebar.radio(
    "请选择功能模块",
    ["项目介绍", "专业数据分析", "成绩预测"],
    index=0
)

# 界面渲染
if page == "项目介绍":
    show_project_intro()
elif page == "专业数据分析":
    show_major_analysis()
elif page == "成绩预测":
    show_score_prediction()
