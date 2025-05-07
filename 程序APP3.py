import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 确保plotly也能显示中文
import plotly.io as pio
pio.templates.default = "simple_white"

# 设置页面配置
st.set_page_config(
    page_title="Gastric Cancer Survival Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'SimHei', 'Times New Roman', serif;
        padding: 1.5rem 0;
        border-bottom: 2px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-family: 'SimHei', 'Times New Roman', serif;
    }
    .description {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    .feature-section {
        padding: 1.5rem;
        background-color: #F9FAFB;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .result-section {
        padding: 2rem;
        background-color: #F0F9FF;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 2rem;
        border: 1px solid #93C5FD;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #6B7280;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 0.5rem;
        border: none;
        margin-top: 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
    /* 改善小型设备上的响应式布局 */
    @media (max-width: 1200px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 加载保存的随机森林模型
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf.pkl')
        # 添加模型信息
        if hasattr(model, 'n_features_in_'):
            st.session_state['model_n_features'] = model.n_features_in_
            st.session_state['model_feature_names'] = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        return model
    except Exception as e:
        st.error(f"⚠️ 模型文件 'rf.pkl' 加载错误: {str(e)}。请确保模型文件在正确的位置。")
        return None

model = load_model()

# 添加调试信息
if model is not None and hasattr(model, 'n_features_in_'):
    st.sidebar.write(f"模型期望特征数量: {model.n_features_in_}")
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        st.sidebar.write("模型期望特征:", expected_features)

# 特征范围定义
feature_ranges = {
    "Intraoperative Blood Loss": {"type": "numerical", "min": 0.000, "max": 800.000, "default": 50, 
                                 "description": "手术期间的出血量 (ml)", "unit": "ml"},
    "CEA": {"type": "numerical", "min": 0, "max": 150.000, "default": 8.68, 
           "description": "癌胚抗原水平", "unit": "ng/ml"},
    "Albumin": {"type": "numerical", "min": 1.0, "max": 80.0, "default": 38.60, 
               "description": "血清白蛋白水平", "unit": "g/L"},
    "TNM Stage": {"type": "categorical", "options": [1, 2, 3, 4], "default": 2, 
                 "description": "肿瘤分期", "unit": ""},
    "Age": {"type": "numerical", "min": 25, "max": 90, "default": 76, 
           "description": "患者年龄", "unit": "岁"},
    "Max Tumor Diameter": {"type": "numerical", "min": 0.2, "max": 20, "default": 4, 
                          "description": "肿瘤最大直径", "unit": "cm"},
    "Lymphovascular Invasion": {"type": "categorical", "options": [0, 1], "default": 1, 
                              "description": "淋巴血管侵犯 (0=否, 1=是)", "unit": ""},
}

# 特征顺序定义 - 确保与模型训练时的顺序一致
# 如果模型有feature_names_in_属性，使用它来定义特征顺序
if model is not None and hasattr(model, 'feature_names_in_'):
    feature_input_order = list(model.feature_names_in_)
    feature_ranges_ordered = {}
    for feature in feature_input_order:
        if feature in feature_ranges:
            feature_ranges_ordered[feature] = feature_ranges[feature]
        else:
            # 模型需要但UI中没有定义的特征
            st.sidebar.warning(f"模型要求特征 '{feature}' 但在UI中未定义")
    
    # 检查UI中定义但模型不需要的特征
    for feature in feature_ranges:
        if feature not in feature_input_order:
            st.sidebar.warning(f"UI中定义的特征 '{feature}' 不在模型要求的特征中")
    
    # 使用排序后的特征字典
    feature_ranges = feature_ranges_ordered
else:
    # 如果模型没有feature_names_in_属性，使用原来的顺序
    feature_input_order = list(feature_ranges.keys())

# 创建二级布局 - 用于在所有设备上更好地显示内容
st.sidebar.markdown("### 调试信息")
st.sidebar.markdown("模型信息会显示在这里")

# 应用标题和描述
st.markdown('<h1 class="main-header">胃癌术后三年生存预测模型</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="description">
    该模型基于术后患者临床特征，预测胃癌患者术后三年内死亡的概率。
    请输入患者的临床参数，系统将提供预测结果并展示影响预测的关键因素。
</div>
""", unsafe_allow_html=True)

# 创建两列布局，调整比例以优化在大屏幕和小屏幕上的显示
col1, col2 = st.columns([3, 5])

with col1:
    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">患者特征输入</h2>', unsafe_allow_html=True)
    
    # 动态生成输入项
    feature_values = {}
    
    for feature in feature_input_order:
        properties = feature_ranges[feature]
        
        # 显示特征描述 - 根据变量类型生成不同的帮助文本
        if properties["type"] == "numerical":
            help_text = f"{properties['description']} ({properties['min']}-{properties['max']} {properties['unit']})"
            
            # 为数值型变量创建滑块
            value = st.slider(
                label=f"{feature}",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                step=0.1,
                help=help_text
            )
        elif properties["type"] == "categorical":
            # 对于分类变量，只使用描述作为帮助文本
            help_text = f"{properties['description']}"
            
            # 为分类变量创建单选按钮
            if feature == "TNM Stage":
                options_display = {1: "I期", 2: "II期", 3: "III期", 4: "IV期"}
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    format_func=lambda x: options_display[x],
                    help=help_text,
                    horizontal=True
                )
            elif feature == "Lymphovascular Invasion":
                options_display = {0: "否", 1: "是"}
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    format_func=lambda x: options_display[x],
                    help=help_text,
                    horizontal=True
                )
            else:
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    help=help_text,
                    horizontal=True
                )
                
        feature_values[feature] = value
    
    # 预测按钮
    predict_button = st.button("开始预测", help="点击生成预测结果")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if predict_button and model is not None:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">预测结果与解释</h2>', unsafe_allow_html=True)
        
        # 准备模型输入
        features_df = pd.DataFrame([feature_values])
        
        # 确保特征顺序与模型训练时一致
        if hasattr(model, 'feature_names_in_'):
            # 检查是否所有需要的特征都有值
            missing_features = [f for f in model.feature_names_in_ if f not in features_df.columns]
            if missing_features:
                st.error(f"缺少模型所需的特征: {missing_features}")
                st.stop()
            
            # 按模型训练时的特征顺序重排列特征
            features_df = features_df[model.feature_names_in_]
        
        # 转换为numpy数组
        features_array = features_df.values
        
        # 添加调试信息
        st.sidebar.write("输入特征形状:", features_array.shape)
        st.sidebar.write("输入特征列名:", list(features_df.columns))
        
        with st.spinner("计算预测结果..."):
            try:
                # 模型预测
                predicted_class = model.predict(features_array)[0]
                predicted_proba = model.predict_proba(features_array)[0]
                
                # 提取预测的类别概率
                death_probability = predicted_proba[1] * 100  # 假设1表示死亡类
                survival_probability = 100 - death_probability
                
                # 创建概率显示
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = death_probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "三年内死亡风险", 'font': {'size': 24, 'family': 'SimHei'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'green'},
                            {'range': [30, 70], 'color': 'orange'},
                            {'range': [70, 100], 'color': 'red'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': death_probability}}))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="white",
                    font={'family': "SimHei"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 创建风险类别显示
                risk_category = "低风险"
                risk_color = "green"
                if death_probability > 30 and death_probability <= 70:
                    risk_category = "中等风险"
                    risk_color = "orange"
                elif death_probability > 70:
                    risk_category = "高风险"
                    risk_color = "red"
                
                # 显示风险类别
                st.markdown(f"""
                <div style="text-align: center; margin-top: -1rem; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem; font-family: 'SimHei'; color: {risk_color}; font-weight: bold;">
                        {risk_category}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # 显示具体概率数值
                risk_cols = st.columns(2)
                with risk_cols[0]:
                    st.metric(label="三年生存概率", value=f"{survival_probability:.1f}%")
                with risk_cols[1]:
                    st.metric(label="三年死亡风险", value=f"{death_probability:.1f}%")
                    
                # 添加SHAP瀑布图
                st.subheader("特征影响分析")
                
                try:
                    # 计算SHAP值
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(features_df)
                    
                    # 根据SHAP值的形状进行处理
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        # 多分类情况，获取死亡类的SHAP值(通常是索引1)
                        class_idx = 1
                        feature_shap_values = shap_values[class_idx][0]
                        base_value = explainer.expected_value[class_idx]
                    elif len(shap_values.shape) == 3:
                        # 三维数组 (样本数, 特征数, 类别数)
                        class_idx = 1  # 假设索引1是死亡风险类
                        feature_shap_values = shap_values[0, :, class_idx]
                        base_value = explainer.expected_value[class_idx]
                    else:
                        # 其他情况(通常是回归问题)
                        feature_shap_values = shap_values[0]
                        base_value = explainer.expected_value
                    
                    # 创建特征名称和SHAP值的DataFrame
                    feature_names = list(features_df.columns)
                    shap_df = pd.DataFrame({
                        '特征': feature_names,
                        'SHAP值': feature_shap_values,
                        '绝对值': np.abs(feature_shap_values)
                    })
                    
                    # 按SHAP值的绝对值排序
                    shap_df = shap_df.sort_values('绝对值', ascending=False)
                    
                    # 创建瀑布图
                    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
                    
                    # 获取排序后的特征和SHAP值
                    features = shap_df['特征'].tolist()
                    shap_values_list = shap_df['SHAP值'].tolist()
                    
                    # 确定颜色：正值为红色(增加风险)，负值为绿色(降低风险)
                    colors = ['red' if x > 0 else 'green' for x in shap_values_list]
                    
                    # 计算累积效果
                    cumulative = [base_value]
                    for value in shap_values_list:
                        cumulative.append(cumulative[-1] + value)
                    
                    # 绘制瀑布图
                    # 基础值
                    ax.barh(0, base_value, left=0, color='lightgray')
                    ax.text(base_value/2, 0, f'基础值\n{base_value:.3f}', ha='center', va='center')
                    
                    # 每个特征的贡献
                    for i, (feature, value) in enumerate(zip(features, shap_values_list)):
                        start = cumulative[i]
                        ax.barh(i+1, value, left=start, color=colors[i])
                        # 添加特征名称和SHAP值
                        if value > 0:
                            ax.text(start + value/2, i+1, f'{feature}\n+{value:.3f}', ha='center', va='center')
                        else:
                            ax.text(start + value/2, i+1, f'{feature}\n{value:.3f}', ha='center', va='center')
                    
                    # 最终预测
                    final_prediction = cumulative[-1]
                    ax.barh(len(features)+1, final_prediction, left=0, color='blue', alpha=0.6)
                    ax.text(final_prediction/2, len(features)+1, f'预测值\n{final_prediction:.3f}', ha='center', va='center', color='white')
                    
                    # 设置坐标轴
                    ax.set_yticks(range(len(features)+2))
                    ax.set_yticklabels(['基础值'] + features + ['预测值'])
                    ax.set_title('特征对预测的影响(瀑布图)', fontsize=14)
                    ax.set_xlabel('特征贡献(对数几率)', fontsize=12)
                    
                    # 添加网格线和美化
                    ax.grid(axis='x', linestyle='--', alpha=0.3)
                    ax.set_axisbelow(True)
                    
                    # 添加图例说明
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='red', label='增加死亡风险'),
                        Patch(facecolor='green', label='降低死亡风险')
                    ]
                    ax.legend(handles=legend_elements, loc='lower right')
                    
                    # 确保图表整洁
                    plt.tight_layout()
                    
                    # 显示图表
                    st.pyplot(fig)
                    
                    # 添加解释性文字
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                    <p style="font-size: 0.9rem;">
                    <strong>瀑布图解释:</strong><br>
                    - <span style="color:red;">红色</span>: 特征值增加患者的死亡风险<br>
                    - <span style="color:green;">绿色</span>: 特征值降低患者的死亡风险<br>
                    - 图表从基础值开始，显示每个特征的贡献，最终得到预测值<br>
                    - 数值表示每个特征对预测的贡献程度
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"生成SHAP瀑布图时出错: {str(e)}")
                    st.info("无法生成特征影响分析图表，请联系技术支持。")
            except Exception as e:
                st.error(f"预测过程中发生错误: {str(e)}")
                st.warning("请检查输入数据是否与模型期望的特征匹配，或联系开发人员获取支持。")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 添加临床建议部分
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">临床建议</h2>', unsafe_allow_html=True)
        
        # 根据不同风险级别提供建议
        if death_probability <= 30:
            st.markdown("""
            <div style="color: green;">
                <p>⭐ <strong>低风险患者建议:</strong></p>
                <ul>
                    <li>遵循标准的术后随访计划</li>
                    <li>每3-6个月进行一次常规检查</li>
                    <li>保持健康的生活方式和饮食习惯</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif death_probability <= 70:
            st.markdown("""
            <div style="color: orange;">
                <p>⚠️ <strong>中等风险患者建议:</strong></p>
                <ul>
                    <li>更频繁的随访计划，建议每2-3个月一次</li>
                    <li>考虑辅助治疗方案</li>
                    <li>密切监测CEA等肿瘤标志物的变化</li>
                    <li>注意营养支持和生活质量管理</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color: red;">
                <p>🔴 <strong>高风险患者建议:</strong></p>
                <ul>
                    <li>更积极的随访计划，建议每1-2个月一次</li>
                    <li>考虑更强化的辅助治疗方案</li>
                    <li>密切监测可能的复发和转移迹象</li>
                    <li>增强营养支持和症状管理</li>
                    <li>考虑多学科团队会诊</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # 显示应用说明和使用指南
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">模型说明</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <p style="font-family: 'Times New Roman'; font-size: 1.1rem;">
            本预测模型基于随机森林算法构建，通过分析胃癌患者的关键临床特征，预测术后三年内的死亡风险。
        </p>
        
        <p style="font-family: 'Times New Roman'; font-size: 1.1rem;">
            <strong>模型特征包括:</strong>
        </p>
        <ul style="font-family: 'Times New Roman'; font-size: 1.1rem;">
            <li><strong>年龄</strong>: 患者年龄是影响胃癌预后的重要因素</li>
            <li><strong>TNM分期</strong>: 描述肿瘤大小、淋巴结侵犯和远处转移情况</li>
            <li><strong>肿瘤直径</strong>: 肿瘤的最大直径</li>
            <li><strong>血清白蛋白</strong>: 反映患者的营养状况</li>
            <li><strong>CEA</strong>: 癌胚抗原，是一种常用的肿瘤标志物</li>
            <li><strong>淋巴血管侵犯</strong>: 指肿瘤是否侵入淋巴或血管</li>
            <li><strong>术中出血量</strong>: 反映手术复杂性和患者耐受性</li>
        </ul>
        
        <p style="font-family: 'Times New Roman'; font-size: 1.1rem; margin-top: 1rem;">
            <strong>使用指南:</strong> 在左侧填写患者的临床参数，然后点击"开始预测"按钮获取结果。系统将生成死亡风险预测以及每个特征的影响程度分析。
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 显示示例案例
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">典型案例分析</h2>', unsafe_allow_html=True)
        
        # 创建示例数据
        case_data = {
            "案例": ["低风险案例", "中风险案例", "高风险案例"],
            "年龄": [55, 68, 76],
            "TNM分期": ["II期", "III期", "IV期"],
            "肿瘤直径(cm)": [2.5, 4.0, 8.5],
            "CEA": [3.2, 7.5, 25.8],
            "预测生存率": ["92%", "58%", "23%"]
        }
        
        case_df = pd.DataFrame(case_data)
        
        # 使用Streamlit的表格显示
        st.dataframe(
            case_df,
            column_config={
                "案例": st.column_config.TextColumn("案例类型"),
                "年龄": st.column_config.NumberColumn("年龄", format="%d岁"),
                "TNM分期": st.column_config.TextColumn("TNM分期"),
                "肿瘤直径(cm)": st.column_config.NumberColumn("肿瘤直径", format="%.1fcm"),
                "CEA": st.column_config.NumberColumn("CEA", format="%.1fng/ml"),
                "预测生存率": st.column_config.TextColumn("3年生存率", width="medium")
            },
            hide_index=True,
            use_container_width=True
        )
                
        st.markdown('</div>', unsafe_allow_html=True)

# 添加页脚说明
st.markdown("""
<div class="disclaimer">
    <p>📋 免责声明：本预测工具仅供临床医生参考，不能替代专业医疗判断。预测结果应结合患者的完整临床情况进行综合评估。</p>
    <p>© 2023 胃癌术后预测研究团队 | 开发版本 v1.2.0</p>
</div>
""", unsafe_allow_html=True) 