import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------- 基础配置（纯Python） --------------------------
# 仅使用Streamlit原生配置，无任何HTML/CSS
st.set_page_config(
    page_title="学生成绩分析与预测系统",
    page_icon="📊",
    layout="wide"
)

# 中文显示配置（仅修改Matplotlib参数，无样式）
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
except:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 数据与模型加载（纯Python） --------------------------
@st.cache_data
def load_data_and_models():
    # 数据加载
    file_name = "student_data_adjusted_rounded.csv"
    try:
        with open(file_name, 'r', encoding='gbk', errors='replace') as f:
            df = pd.read_csv(f)
        
        # 列名清理（纯字符串处理）
        df.columns = (
            df.columns
            .str.strip()
            .str.replace('（小时）', '', regex=False)
            .str.replace('（', '(', regex=False)
            .str.replace('）', ')', regex=False)
        )
        
        # 数据清洗（纯Pandas操作）
        required_cols = ['学号', '性别', '专业', '每周学习时长', '上课出勤率', '期中考试分数', '作业完成率', '期末考试分数']
        df = df[required_cols].dropna()
        numeric_cols = ['每周学习时长', '上课出勤率', '期中考试分数', '作业完成率', '期末考试分数']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
    except Exception as e:
        st.error(f"数据加载失败：{str(e)}")
        return None, None, None, None

    # 模型加载（纯joblib操作）
    reg_model_filename = "linear_regression_model.pkl"
    clf_model_filename = "random_forest_clf.pkl"
    feature_cols_filename = "clf_feature_cols.pkl"

    reg_model = None
    clf_model = None
    clf_feature_cols = None

    try:
        reg_model = joblib.load(reg_model_filename)
    except Exception as e:
        st.error(f"数值预测模型加载失败：{str(e)}")

    try:
        clf_model = joblib.load(clf_model_filename)
        clf_feature_cols = joblib.load(feature_cols_filename)
    except Exception as e:
        st.error(f"分类预测模型加载失败：{str(e)}")

    return df, reg_model, clf_model, clf_feature_cols

# 初始化数据和模型
student_df, reg_model, clf_model, clf_feature_cols = load_data_and_models()

# -------------------------- 辅助函数（纯Python） --------------------------
def build_clf_input(input_data, base_df_cols):
    """构造分类模型输入特征（纯Pandas操作）"""
    raw_df = pd.DataFrame({
        'gender': [input_data['gender']],
        'major': [input_data['major']],
        'study_hour': [input_data['study_hour']],
        'attendance': [input_data['attendance']],
        'mid_score': [input_data['mid_score']],
        'homework_rate': [input_data['homework_rate']]
    })
    
    # 独热编码+特征补全（纯Pandas操作）
    encoded_df = pd.get_dummies(raw_df)
    for col in base_df_cols:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
    final_df = encoded_df[base_df_cols]
    return final_df

# -------------------------- 界面1：项目介绍（纯Streamlit原生组件） --------------------------
def show_project_intro():
    st.title("学生成绩分析与预测系统")
    st.divider()

    # 分栏布局（Streamlit原生columns）
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

        # 项目目标（Streamlit原生分栏）
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

        # 技术栈（Streamlit原生分栏）
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
        st.image("系统预览图.png", use_container_width=True)

# -------------------------- 界面2：专业数据分析（纯Python/Matplotlib） --------------------------
def show_major_analysis():
    if student_df is None:
        st.warning("⚠️ 数据加载失败，无法进行分析！")
        return
    
    st.title("专业成绩数据分析")
    st.divider()

    # 1. 性别比例分析（纯Matplotlib+Streamlit）
    st.subheader("1. 各专业男女性别比例")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        gender_stats = student_df.groupby('专业')['性别'].value_counts(normalize=True).unstack().fillna(0)
        gender_stats.columns = ['男性比例', '女性比例']
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

    # 2. 学习指标对比（纯Matplotlib+Streamlit）
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

    # 3. 出勤率分析（纯Matplotlib+Streamlit）
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

    # 4. 大数据管理专项分析（纯Python/Matplotlib）
    st.subheader("4. 大数据管理专业专项分析")
    if '大数据管理' in student_df['专业'].unique():
        bd_df = student_df[student_df['专业'] == '大数据管理']
        # 指标卡片（Streamlit原生metric）
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("平均出勤率", f"{bd_df['上课出勤率'].mean():.1%}")
        with col2:
            st.metric("平均期末分数", f"{bd_df['期末考试分数'].mean():.1f}")
        with col3:
            st.metric("通过率", f"{(bd_df['期末考试分数'] >= 60).mean():.1%}")
        with col4:
            st.metric("平均学习时长", f"{bd_df['每周学习时长'].mean():.1f}小时")
        
        # 分布图表（纯Matplotlib）
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

# -------------------------- 界面3：成绩预测（纯Python/Streamlit） --------------------------
def show_score_prediction():
    if reg_model is None or clf_model is None or clf_feature_cols is None:
        st.warning("⚠️ 模型加载不完整，无法进行成绩预测！")
        return
    
    st.title("期末成绩预测")
    st.write("📝 输入学生信息，系统将预测期末成绩并匹配对应等级表情包")
    st.divider()

    # 输入表单（Streamlit原生组件）
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("学号", value="2024001")
        gender = st.selectbox("性别", ["男", "女"], index=0)
        major = st.selectbox("专业", student_df['专业'].unique(), index=0)
        submit_btn = st.button("🚀 预测期末成绩", type="primary")
    with col2:
        study_hour = st.slider("每周学习时长（小时）", 0.0, 50.0, 15.0, 0.1)
        attendance = st.slider("上课出勤率", 0.0, 1.0, 0.9, 0.01)
        mid_score = st.slider("期中考试分数", 0.0, 100.0, 70.0, 0.1)
        homework_rate = st.slider("作业完成率", 0.0, 1.0, 0.95, 0.01)

    # 预测逻辑（纯Python/sklearn）
    if submit_btn:
        try:
            # 数值预测
            reg_input = np.array([[study_hour, attendance, mid_score, homework_rate]])
            pred_score = reg_model.predict(reg_input)[0]
            pred_score = np.clip(pred_score, 0, 100)

            # 分类预测
            input_data = {
                'gender': gender,
                'major': major,
                'study_hour': study_hour,
                'attendance': attendance,
                'mid_score': mid_score,
                'homework_rate': homework_rate
            }
            clf_input = build_clf_input(input_data, clf_feature_cols)
            pred_clf = clf_model.predict(clf_input)[0]

            # 及格状态判断
            if pred_score >= 60:
                pred_clf_label = "及格"
                delta_text = "达标"
            else:
                pred_clf_label = "不及格"
                delta_text = "需提升"

            # 结果展示（Streamlit原生组件）
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

            # 等级匹配（纯Python逻辑）
            st.subheader("📊 成绩等级匹配")
            emoji_paths = {
                "不及格": "未及格.PNG",
                "及格": "及格.PNG",
                "良好": "良好.PNG",
                "优秀": "优秀.PNG"
            }

            if pred_score < 60:
                level = "不及格"
                st.image(emoji_paths[level], width=150)
            elif 60 <= pred_score < 70:
                level = "及格"
                st.image(emoji_paths[level], width=150)
            elif 70 <= pred_score < 90:
                level = "良好"
                st.image(emoji_paths[level], width=150)
            else:
                level = "优秀"
                st.image(emoji_paths[level], width=150)

        except Exception as e:
            st.error(f"❌ 预测失败：{str(e)}")
            st.info("提示：请检查模型特征与输入特征是否匹配，或数据文件路径是否正确")

# -------------------------- 导航菜单（Streamlit原生组件） --------------------------
st.sidebar.title("📚 系统导航")
page = st.sidebar.radio(
    "请选择功能模块",
    ["项目介绍", "专业数据分析", "成绩预测"],
    index=0
)

# 界面渲染（纯Python分支逻辑）
if page == "项目介绍":
    show_project_intro()
elif page == "专业数据分析":
    show_major_analysis()
elif page == "成绩预测":
    show_score_prediction()
