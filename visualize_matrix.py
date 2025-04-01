import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re

def analyze_and_visualize_matrix(folder_paths, output_folder=None):
    """只生成模型交叉分析矩阵"""
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]  # 如果是单个文件夹，转换为列表
    
    if output_folder is None:
        # 默认输出到第一个文件夹的上级目录的matrix_output子文件夹
        parent_dir = os.path.dirname(os.path.abspath(folder_paths[0]))
        output_folder = os.path.join(parent_dir, 'matrix_output')
    
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
    
    # 模型名称映射
    model_name_mapping = {
        'qwen': 'Qwen',
        'claude': 'Claude',
        'gemini': 'Gemini',
        'llama': 'Llama',
        'mistral': 'Mistral',
        'deepseek': 'Deepseek',
        'anthropic': 'Claude',
        'meta-llama': 'Llama',
        'google': 'Gemini'
    }
    
    # 指定哪些模型是数据生成者（从文件夹名中确定）
    generator_models = set()
    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path)
        
        # 提取文件夹名称中的模型
        for key, value in model_name_mapping.items():
            if key.lower() in folder_name.lower():
                generator_models.add(value)
                break
    
    print(f"Identified generator models from folder names: {', '.join(generator_models)}")
    
    for file_path in analysis_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 基本文件信息
            file_name = os.path.basename(file_path)
            
            # 从JSON数据获取分析器模型名称
            full_model_name = data.get('model_name', 'unknown')
            
            # 简化分析器模型名称
            analyzer_model = 'Unknown'
            for key, value in model_name_mapping.items():
                if key in full_model_name.lower():
                    analyzer_model = value
                    break
            
            # 如果还是未知，尝试从文件名获取分析器模型
            if analyzer_model == 'Unknown':
                # 尝试从文件名格式提取：balanced20Posts_[confidence]_[topic]_[generator]_[analyzer]_analysis.json
                parts = file_name.split('_')
                if len(parts) >= 5:  # 确保文件名有足够的部分
                    # 分析器通常是最后一个下划线前的部分
                    for i in range(len(parts) - 2, 0, -1):  # 从倒数第二个部分向前查找
                        for key, value in model_name_mapping.items():
                            if key.lower() in parts[i].lower():
                                analyzer_model = value
                                break
                        if analyzer_model != 'Unknown':
                            break
            
            # 从文件名中识别生成器模型
            generator_model = 'Unknown'
            
            # 尝试从文件名格式提取：balanced20Posts_[confidence]_[topic]_[generator]_[analyzer]_analysis.json
            match = re.search(r'(\w+)-(\w+)-(\w+)(?:-\w+)?_(\w+)_(\w+)', file_name)
            if match:
                generator_name = match.group(1)
                for key, value in model_name_mapping.items():
                    if key.lower() in generator_name.lower():
                        generator_model = value
                        break
            
            # 如果从文件名中无法提取，尝试使用文件夹名称
            if generator_model == 'Unknown':
                folder_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                for gen_model in generator_models:
                    if gen_model.lower() in folder_name.lower():
                        generator_model = gen_model
                        break
            
            # 提取指标
            metrics = data.get('metrics', {})
            accuracy = metrics.get('overall_accuracy', 0)
            
            # 添加到文件数据列表
            file_entry = {
                'file': file_name,
                'analyzer_model': analyzer_model,
                'generator_model': generator_model,
                'accuracy': accuracy
            }
            
            file_data.append(file_entry)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # 创建数据框
    df = pd.DataFrame(file_data)
    
    if df.empty:
        print("No valid data found in the specified folders")
        return
    
    # 过滤掉无法识别的模型
    df = df[(df['analyzer_model'] != 'Unknown') & (df['generator_model'] != 'Unknown')]
    
    if df.empty:
        print("No valid data after filtering unknown models")
        return
    
    # 如果generator_models为空，则从数据中提取所有唯一的生成模型
    if not generator_models:
        generator_models = set(df['generator_model'].unique())
    
    print(f"Analyzer models found: {', '.join(df['analyzer_model'].unique())}")
    print(f"Generator models to include in matrix: {', '.join(generator_models)}")
    
    # 创建模型交叉分析矩阵
    create_model_cross_analysis_matrix(df, output_folder, generator_models)
    
    print(f"Matrix visualizations created in {output_folder}")

def create_model_cross_analysis_matrix(df, output_folder, generator_models):
    """创建模型交叉分析矩阵图"""
    # 确保数据中有分析模型和生成模型信息
    df_cross = df.dropna(subset=['analyzer_model', 'generator_model'])
    if df_cross.empty:
        print("No data with both analyzer and generator model information available")
        return
    
    # 计算每个分析模型对每个生成模型的平均准确率
    cross_accuracy = df_cross.groupby(['analyzer_model', 'generator_model'])['accuracy'].mean().reset_index()
    
    # 创建交叉矩阵
    try:
        # 透视表转换
        pivot_data = cross_accuracy.pivot_table(
            values='accuracy', 
            index='analyzer_model', 
            columns='generator_model',
            aggfunc='mean'
        )
        
        # 确保只包含指定的生成模型
        for gen_model in generator_models:
            if gen_model not in pivot_data.columns:
                pivot_data[gen_model] = np.nan
        
        # 只保留指定的生成模型列
        pivot_data = pivot_data[[col for col in pivot_data.columns if col in generator_models]]
        
        # 绘制矩阵热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100, 
                    linewidths=0.5, mask=pivot_data.isna())
        plt.title('Model Cross-Analysis Accuracy Matrix\n(Rows: Analyzer Models, Columns: Generator Models)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'model_cross_analysis_matrix.png'))
        plt.close()
        
        # 将数据保存为CSV文件，便于进一步分析
        pivot_data.to_csv(os.path.join(output_folder, 'model_cross_analysis_matrix.csv'))
        
        # 计算分析模型对生成模型的相对表现
        # 即每个分析模型对各生成模型的准确率与该生成模型的平均准确率的偏差
        gen_model_avg = cross_accuracy.groupby('generator_model')['accuracy'].mean()
        
        relative_performance = []
        for _, row in cross_accuracy.iterrows():
            analyzer = row['analyzer_model']
            generator = row['generator_model']
            if generator not in generator_models:
                continue
                
            accuracy = row['accuracy']
            gen_avg = gen_model_avg[generator]
            
            relative_performance.append({
                'analyzer_model': analyzer,
                'generator_model': generator,
                'relative_accuracy': accuracy - gen_avg
            })
        
        rel_perf_df = pd.DataFrame(relative_performance)
        
        # 绘制相对表现矩阵
        try:
            pivot_rel = rel_perf_df.pivot_table(
                values='relative_accuracy', 
                index='analyzer_model', 
                columns='generator_model',
                aggfunc='mean'
            )
            
            # 只保留指定的生成模型列
            for gen_model in generator_models:
                if gen_model not in pivot_rel.columns:
                    pivot_rel[gen_model] = np.nan
                    
            pivot_rel = pivot_rel[[col for col in pivot_rel.columns if col in generator_models]]
            
            # 使用发散色谱
            plt.figure(figsize=(12, 10))
            sns.heatmap(pivot_rel, annot=True, fmt='.1f', cmap='RdBu_r', 
                        center=0, vmin=-20, vmax=20, linewidths=0.5,
                        mask=pivot_rel.isna())
            plt.title('Relative Performance Matrix\n(How much better/worse each analyzer performs compared to the average for each generator)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'model_relative_performance_matrix.png'))
            plt.close()
            
            # 保存相对表现数据
            pivot_rel.to_csv(os.path.join(output_folder, 'model_relative_performance_matrix.csv'))
            
        except Exception as e:
            print(f"Could not create relative performance matrix: {e}")
        
        # 创建分析器性能比较图
        try:
            analyzer_avg = cross_accuracy[cross_accuracy['generator_model'].isin(generator_models)]
            analyzer_avg = analyzer_avg.groupby('analyzer_model')['accuracy'].mean().reset_index()
            
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x='analyzer_model', y='accuracy', data=analyzer_avg)
            
            # 添加数值标签
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold')
            
            plt.title('Average Accuracy by Analyzer Model\n(Across Generator Models: ' + ', '.join(generator_models) + ')')
            plt.xlabel('Analyzer Model')
            plt.ylabel('Average Accuracy (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'analyzer_performance_comparison.png'))
            plt.close()
            
            # 保存分析器平均性能数据
            analyzer_avg.to_csv(os.path.join(output_folder, 'analyzer_performance_comparison.csv'))
        
        except Exception as e:
            print(f"Could not create analyzer performance comparison: {e}")
        
    except Exception as e:
        print(f"Could not create cross-analysis matrix: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate model cross-analysis matrix')
    parser.add_argument('folder_paths', nargs='+', help='Folders containing analysis files')
    parser.add_argument('--output', help='Output folder for matrix visualizations')
    
    args = parser.parse_args()
    
    analyze_and_visualize_matrix(args.folder_paths, args.output)