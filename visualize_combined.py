import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse

def analyze_and_visualize_political_data(folder_paths, output_folder=None):
    """分析并可视化政治立场分类结果，支持多个文件夹"""
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]  # 如果是单个文件夹，转换为列表
    
    if output_folder is None:
        # 默认输出到第一个文件夹的上级目录的visualizations子文件夹
        parent_dir = os.path.dirname(os.path.abspath(folder_paths[0]))
        output_folder = os.path.join(parent_dir, 'visualizations')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 收集所有分析文件
    analysis_files = []
    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('_analysis.json'):
                    analysis_files.append(os.path.join(root, file))
    
    print(f"Found {len(analysis_files)} analysis files across {len(folder_paths)} folders")
    
    # 提取文件数据
    file_data = []
    model_data = {}
    topic_confidence_data = {}
    
    for file_path in analysis_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 基本文件信息
            file_name = os.path.basename(file_path)
            
            # 从文件路径获取模型名称
            folder_name = os.path.basename(os.path.dirname(file_path))
            if "Deepseek" in folder_name:
                model = "Deepseek"
            elif "Gemini" in folder_name:
                model = "Gemini"
            elif "Mistral" in folder_name:
                model = "Mistral"
            else:
                model = data.get('model_name', 'unknown')
            
            # 提取主题和信心度
            topic = None
            confidence = None
            
            if 'high_confidence' in file_name:
                confidence = 'high'
            elif 'low_confidence' in file_name:
                confidence = 'low'
            
            # 尝试从文件名中提取主题
            parts = file_name.split('_confidence_')
            if len(parts) > 1:
                topic_part = parts[1].split('_')[0]
                topic = topic_part
            
            # 提取指标
            metrics = data['metrics']
            accuracy = metrics['overall_accuracy']
            
            # 修复F1分数
            for orientation in ['right', 'left', 'unknown']:
                orient_metrics = metrics[orientation]
                precision = orient_metrics['precision']
                recall = orient_metrics['recall']
                
                if (precision + recall) > 0:
                    corrected_f1 = 2 * precision * recall / (precision + recall)
                else:
                    corrected_f1 = 0
                
                orient_metrics['corrected_f1'] = corrected_f1
            
            # 添加到文件数据列表
            file_entry = {
                'file': file_name,
                'model': model,
                'topic': topic,
                'confidence': confidence,
                'accuracy': accuracy,
                'right_precision': metrics['right']['precision'],
                'right_recall': metrics['right']['recall'],
                'right_f1': metrics['right']['corrected_f1'],
                'left_precision': metrics['left']['precision'],
                'left_recall': metrics['left']['recall'],
                'left_f1': metrics['left']['corrected_f1'],
                'unknown_precision': metrics['unknown']['precision'],
                'unknown_recall': metrics['unknown']['recall'],
                'unknown_f1': metrics['unknown']['corrected_f1']
            }
            
            file_data.append(file_entry)
            
            # 添加到模型数据
            if model not in model_data:
                model_data[model] = []
            model_data[model].append(file_entry)
            
            # 添加到主题-信心度数据
            if topic and confidence:
                key = (topic, confidence)
                if key not in topic_confidence_data:
                    topic_confidence_data[key] = []
                topic_confidence_data[key].append(file_entry)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # 创建数据框
    df = pd.DataFrame(file_data)
    
    if df.empty:
        print("No valid data found in the specified folders")
        return
    
    print(f"Analyzing data for models: {', '.join(df['model'].unique())}")
    
    # 1. 模型性能比较
    visualize_model_performance(df, output_folder)
    
    # 2. 主题分析
    visualize_topics(df, output_folder)
    
    # 3. 信心度分析
    visualize_confidence_levels(df, output_folder)
    
    # 4. 主题-信心度热力图
    create_topic_confidence_heatmap(df, output_folder)
    
    print(f"Visualizations created in {output_folder}")

def add_value_labels(ax):
    """在柱状图上添加数值标签"""
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() 
        ax.annotate(f'{height:.1f}', 
                    (x + width/2, height), 
                    ha='center', 
                    va='bottom', 
                    fontweight='bold',
                    fontsize=9)

def visualize_model_performance(df, output_folder):
    """比较不同模型的性能"""
    # 1. 模型平均准确率
    model_accuracy = df.groupby('model')['accuracy'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='model', y='accuracy', data=model_accuracy)
    add_value_labels(ax)
    plt.title('Average Accuracy by Model')
    plt.xlabel('Model')
    plt.ylabel('Average Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'model_accuracy_comparison.png'))
    plt.close()
    
    # 2. 模型F1分数比较
    model_metrics = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_metrics.append({
            'model': model,
            'right_f1': model_df['right_f1'].mean(),
            'left_f1': model_df['left_f1'].mean(),
            'unknown_f1': model_df['unknown_f1'].mean()
        })
    
    model_metrics_df = pd.DataFrame(model_metrics)
    
    # 为F1指标创建更好的标签
    model_metrics_long = pd.melt(model_metrics_df, 
                                id_vars=['model'], 
                                value_vars=['right_f1', 'left_f1', 'unknown_f1'],
                                var_name='orientation', 
                                value_name='f1_score')
    
    # 创建更易读的标签
    model_metrics_long['orientation'] = model_metrics_long['orientation'].map({
        'right_f1': 'Right', 
        'left_f1': 'Left', 
        'unknown_f1': 'Unknown'
    })
    
    # 增加图表宽度以便在右侧有足够空间放置图例
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(x='model', y='f1_score', hue='orientation', data=model_metrics_long)
    
    # 为每组柱状图添加标签
    for i, group in enumerate(ax.patches):
        height = group.get_height()
        if not np.isnan(height):  # 确保数值不是NaN
            ax.text(
                group.get_x() + group.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                fontsize=8,
                fontweight='bold'
            )
    
    plt.title('Average F1 Score by Model and Orientation')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    
    # 将图例放在图表外部右侧
    plt.legend(title='Orientation', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局，确保图例完全可见
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'model_f1_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # 3. 模型召回率比较
    model_recall = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_recall.append({
            'model': model,
            'right_recall': model_df['right_recall'].mean(),
            'left_recall': model_df['left_recall'].mean(),
            'unknown_recall': model_df['unknown_recall'].mean()
        })
    
    model_recall_df = pd.DataFrame(model_recall)
    
    # 为召回率指标创建更好的标签
    model_recall_long = pd.melt(model_recall_df, 
                                id_vars=['model'], 
                                value_vars=['right_recall', 'left_recall', 'unknown_recall'],
                                var_name='orientation', 
                                value_name='recall')
    
    # 创建更易读的标签
    model_recall_long['orientation'] = model_recall_long['orientation'].map({
        'right_recall': 'Right', 
        'left_recall': 'Left', 
        'unknown_recall': 'Unknown'
    })
    
    # 增加图表宽度以便在右侧有足够空间放置图例
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(x='model', y='recall', hue='orientation', data=model_recall_long)
    
    # 为每组柱状图添加标签
    for i, group in enumerate(ax.patches):
        height = group.get_height()
        if not np.isnan(height):  # 确保数值不是NaN
            ax.text(
                group.get_x() + group.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                fontsize=8,
                fontweight='bold'
            )
    
    plt.title('Average Recall by Model and Orientation')
    plt.xlabel('Model')
    plt.ylabel('Recall')
    plt.xticks(rotation=45)
    
    # 将图例放在图表外部右侧
    plt.legend(title='Orientation', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局，确保图例完全可见
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'model_recall_comparison.png'), bbox_inches='tight')
    plt.close()

def visualize_topics(df, output_folder):
    """分析不同主题的表现"""
    if 'topic' not in df.columns or df['topic'].isna().all():
        print("No topic information available")
        return
    
    # 1. 主题准确率
    topic_accuracy = df.groupby('topic')['accuracy'].mean().reset_index()
    
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x='topic', y='accuracy', data=topic_accuracy)
    add_value_labels(ax)
    plt.title('Average Accuracy by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Average Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'topic_accuracy_comparison.png'))
    plt.close()
    
    # 2. 主题分类指标
    topic_metrics = []
    for topic in df['topic'].dropna().unique():
        topic_df = df[df['topic'] == topic]
        topic_metrics.append({
            'topic': topic,
            'right_precision': topic_df['right_precision'].mean(),
            'right_recall': topic_df['right_recall'].mean(),
            'left_precision': topic_df['left_precision'].mean(),
            'left_recall': topic_df['left_recall'].mean()
        })
    
    topic_metrics_df = pd.DataFrame(topic_metrics)
    topic_metrics_long = pd.melt(topic_metrics_df, 
                                id_vars=['topic'], 
                                value_vars=['right_precision', 'right_recall', 'left_precision', 'left_recall'],
                                var_name='metric', 
                                value_name='value')
    
    # 添加更易读的标签
    topic_metrics_long['metric'] = topic_metrics_long['metric'].map({
        'right_precision': 'Right Precision', 
        'right_recall': 'Right Recall', 
        'left_precision': 'Left Precision', 
        'left_recall': 'Left Recall'
    })
    
    # 增加图表宽度以便右侧有足够空间放置图例
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(x='topic', y='value', hue='metric', data=topic_metrics_long)
    
    # 为每组柱状图添加标签
    for i, group in enumerate(ax.patches):
        height = group.get_height()
        if not np.isnan(height):  # 确保数值不是NaN
            ax.text(
                group.get_x() + group.get_width()/2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                fontsize=8,
                fontweight='bold'
            )
    
    plt.title('Average Metrics by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Score (%)')
    plt.xticks(rotation=45)
    
    # 将图例放在图表外部右侧
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局，确保图例完全可见
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'topic_metrics_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # # 3. 按模型和主题分析
    # model_topic_accuracy = df.groupby(['model', 'topic'])['accuracy'].mean().reset_index()
    
    # plt.figure(figsize=(16, 8))
    # ax = sns.barplot(x='topic', y='accuracy', hue='model', data=model_topic_accuracy)
    
    # # 为分组柱状图添加标签
    # for i, model in enumerate(df['model'].unique()):
    #     for j, topic in enumerate(df['topic'].dropna().unique()):
    #         subset = model_topic_accuracy[(model_topic_accuracy['model']==model) & 
    #                                      (model_topic_accuracy['topic']==topic)]
    #         if not subset.empty:
    #             value = subset['accuracy'].values[0]
    #             x = j + (i - 0.5) * 0.25  # 调整x位置以适应不同分组的柱状图
    #             plt.text(x, value + 0.01, f'{value:.1f}', ha='center', fontsize=7, fontweight='bold')
    
    # plt.title('Average Accuracy by Model and Topic')
    # plt.xlabel('Topic')
    # plt.ylabel('Average Accuracy (%)')
    # plt.xticks(rotation=45)
    # plt.legend(title='Model')
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, 'model_topic_comparison.png'))
    # plt.close()
    
    # 3. 替換為熱力圖呈現模型 × 主題的準確率
    model_topic_accuracy = df.groupby(['model', 'topic'])['accuracy'].mean().reset_index()
    pivot_table = model_topic_accuracy.pivot(index='model', columns='topic', values='accuracy')

    # 使用紅白綠配色（數值越高越綠）
    custom_cmap = sns.color_palette("RdYlGn", as_cmap=True)

    plt.figure(figsize=(18, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap=custom_cmap, linewidths=0.5, cbar_kws={'label': 'Accuracy (%)'})
    plt.title('Model × Topic Accuracy Heatmap')
    plt.xlabel('Topic')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'model_topic_comparison.png'))
    plt.close()

def visualize_confidence_levels(df, output_folder):
    """分析不同信心度的表现"""
    if 'confidence' not in df.columns or df['confidence'].isna().all():
        print("No confidence level information available")
        return
    
    # 1. 信心度准确率对比
    confidence_accuracy = df.groupby('confidence')['accuracy'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='confidence', y='accuracy', data=confidence_accuracy)
    add_value_labels(ax)
    plt.title('Average Accuracy by Confidence Level')
    plt.xlabel('Confidence Level')
    plt.ylabel('Average Accuracy (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confidence_accuracy_comparison.png'))
    plt.close()
    
    # 2. 按模型和信心度分组
    model_confidence = df.groupby(['model', 'confidence'])['accuracy'].mean().reset_index()
    
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x='model', y='accuracy', hue='confidence', data=model_confidence)
    
    # 为分组柱状图添加标签
    for i, confidence in enumerate(model_confidence['confidence'].unique()):
        for j, model in enumerate(model_confidence['model'].unique()):
            subset = model_confidence[(model_confidence['model']==model) & 
                                     (model_confidence['confidence']==confidence)]
            if not subset.empty:
                value = subset['accuracy'].values[0]
                x = j + (i - 0.5) * 0.4  # 调整x位置以适应不同分组的柱状图
                plt.text(x, value + 0.01, f'{value:.1f}', ha='center', fontsize=8, fontweight='bold')
    
    plt.title('Average Accuracy by Model and Confidence Level')
    plt.xlabel('Model')
    plt.ylabel('Average Accuracy (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Confidence')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'model_confidence_comparison.png'))
    plt.close()

def create_topic_confidence_heatmap(df, output_folder):
    """创建主题-信心度热力图"""
    if 'topic' not in df.columns or 'confidence' not in df.columns:
        print("No topic or confidence information available")
        return
    
    # 准备热力图数据
    pivot_data = df.pivot_table(
        values='accuracy', 
        index='topic', 
        columns='confidence',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', linewidths=0.5)
    plt.title('Average Accuracy by Topic and Confidence Level')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'topic_confidence_heatmap.png'))
    plt.close()
    
    # 为每个模型创建热力图
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        try:
            pivot_data = model_df.pivot_table(
                values='accuracy', 
                index='topic', 
                columns='confidence',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', linewidths=0.5)
            plt.title(f'Average Accuracy by Topic and Confidence Level - {model}')
            plt.tight_layout()
            
            # Replace / with _ in model name for filename
            safe_model_name = model.replace('/', '_')
            plt.savefig(os.path.join(output_folder, f'{safe_model_name}_topic_confidence_heatmap.png'))
            plt.close()
            
        except Exception as e:
            print(f"Could not create heatmap for model: {model}")
            print(f"Error details: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize political stance analysis results')
    parser.add_argument('folder_paths', nargs='+', help='Folders containing analysis files')
    parser.add_argument('--output', help='Output folder for visualizations')
    
    args = parser.parse_args()
    
    analyze_and_visualize_political_data(args.folder_paths, args.output)