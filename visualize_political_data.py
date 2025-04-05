import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse

def analyze_and_visualize_political_data(folder_path, output_folder=None):
    """Analyze and visualize political stance classification results"""
    if output_folder is None:
        output_folder = os.path.join(folder_path, 'visualizations')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all analysis files
    analysis_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('_analysis.json'):
                analysis_files.append(os.path.join(root, file))
    
    print(f"Found {len(analysis_files)} analysis files")
    
    # Extract file data
    file_data = []
    model_data = {}
    topic_confidence_data = {}
    
    for file_path in analysis_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Basic file information
            file_name = os.path.basename(file_path)
            model = data.get('model_name', 'unknown')
            
            # Extract topic and confidence
            topic = None
            confidence = None
            
            if 'high_confidence' in file_name:
                confidence = 'high'
            elif 'low_confidence' in file_name:
                confidence = 'low'
            
            # Try to extract topic from filename
            parts = file_name.split('_confidence_')
            if len(parts) > 1:
                topic_part = parts[1].split('_')[0]
                topic = topic_part
            
            # Extract metrics
            metrics = data['metrics']
            accuracy = metrics['overall_accuracy']
            
            # Fix F1 scores
            for orientation in ['right', 'left', 'unknown']:
                orient_metrics = metrics[orientation]
                precision = orient_metrics['precision']
                recall = orient_metrics['recall']
                
                if (precision + recall) > 0:
                    corrected_f1 = 2 * precision * recall / (precision + recall)
                else:
                    corrected_f1 = 0
                
                orient_metrics['corrected_f1'] = corrected_f1
            
            # Add to file data list
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
            
            # Add to model data
            if model not in model_data:
                model_data[model] = []
            model_data[model].append(file_entry)
            
            # Add to topic-confidence data
            if topic and confidence:
                key = (topic, confidence)
                if key not in topic_confidence_data:
                    topic_confidence_data[key] = []
                topic_confidence_data[key].append(file_entry)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create dataframe
    df = pd.DataFrame(file_data)
    
    # 1. Model performance comparison
    visualize_model_performance(df, output_folder)
    
    # 2. Topic analysis
    visualize_topics(df, output_folder)
    
    # 3. Confidence level analysis
    visualize_confidence_levels(df, output_folder)
    
    # 4. Topic-confidence heatmap
    create_topic_confidence_heatmap(df, output_folder)
    
    print(f"Visualizations created in {output_folder}")

def add_value_labels(ax):
    """Add value labels to bar charts"""
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
    """Compare performance of different models"""
    # 1. Model average accuracy
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
    
    # 2. Model F1 score comparison
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
    
    # Create better labels for F1 metrics
    model_metrics_long = pd.melt(model_metrics_df, 
                                id_vars=['model'], 
                                value_vars=['right_f1', 'left_f1', 'unknown_f1'],
                                var_name='orientation', 
                                value_name='f1_score')
    
    # Create more readable labels
    model_metrics_long['orientation'] = model_metrics_long['orientation'].map({
        'right_f1': 'Right', 
        'left_f1': 'Left', 
        'unknown_f1': 'Unknown'
    })
    
    # Increase chart width to have enough space for the legend on the right
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(x='model', y='f1_score', hue='orientation', data=model_metrics_long)
    
    # Add labels for each group of bars
    for i, group in enumerate(ax.patches):
        height = group.get_height()
        if not np.isnan(height):  # Ensure value is not NaN
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
    
    # Place legend outside the chart on the right
    plt.legend(title='Orientation', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to ensure the legend is fully visible
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'model_f1_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Model recall comparison
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
    
    # Create better labels for recall metrics
    model_recall_long = pd.melt(model_recall_df, 
                                id_vars=['model'], 
                                value_vars=['right_recall', 'left_recall', 'unknown_recall'],
                                var_name='orientation', 
                                value_name='recall')
    
    # Create more readable labels
    model_recall_long['orientation'] = model_recall_long['orientation'].map({
        'right_recall': 'Right', 
        'left_recall': 'Left', 
        'unknown_recall': 'Unknown'
    })
    
    # Increase chart width to have enough space for the legend on the right
    plt.figure(figsize=(16, 7))
    ax = sns.barplot(x='model', y='recall', hue='orientation', data=model_recall_long)
    
    # Add labels for each group of bars
    for i, group in enumerate(ax.patches):
        height = group.get_height()
        if not np.isnan(height):  # Ensure value is not NaN
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
    
    # Place legend outside the chart on the right
    plt.legend(title='Orientation', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to ensure the legend is fully visible
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'model_recall_comparison.png'), bbox_inches='tight')
    plt.close()

def visualize_topics(df, output_folder):
    """Analyze performance across different topics"""
    if 'topic' not in df.columns or df['topic'].isna().all():
        print("No topic information available")
        return
    
    # 1. Topic accuracy
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
    
    # 2. Topic classification metrics
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
    
    # Add more readable labels
    topic_metrics_long['metric'] = topic_metrics_long['metric'].map({
        'right_precision': 'Right Precision', 
        'right_recall': 'Right Recall', 
        'left_precision': 'Left Precision', 
        'left_recall': 'Left Recall'
    })
    
    # Increase chart width to have enough space for the legend on the right
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(x='topic', y='value', hue='metric', data=topic_metrics_long)
    
    # Add labels for each group of bars
    for i, group in enumerate(ax.patches):
        height = group.get_height()
        if not np.isnan(height):  # Ensure value is not NaN
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
    
    # Place legend outside the chart on the right
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to ensure the legend is fully visible
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'topic_metrics_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # # 2. Change to scatter plot (Right vs Left, Precision vs Recall)
    # topic_metrics = []
    # for topic in df['topic'].dropna().unique():
    #     topic_df = df[df['topic'] == topic]
    #     topic_metrics.append({
    #         'topic': topic,
    #         'right_precision': topic_df['right_precision'].mean(),
    #         'right_recall': topic_df['right_recall'].mean(),
    #         'left_precision': topic_df['left_precision'].mean(),
    #         'left_recall': topic_df['left_recall'].mean()
    #     })

    # topic_df = pd.DataFrame(topic_metrics)

    # plt.figure(figsize=(14, 10))

    # # x = y baseline
    # plt.plot([0, 100], [0, 100], linestyle='--', color='gray', linewidth=1, label='x = y baseline')

    # # Plot right-wing points
    # plt.scatter(topic_df['right_precision'], topic_df['right_recall'], color='#1f77b4', label='Right', s=80, alpha=0.8)
    # for i in range(len(topic_df)):
    #     x = topic_df['right_precision'][i]
    #     y = topic_df['right_recall'][i]
    #     plt.text(x + 0.8, y + 0.8, topic_df['topic'][i], fontsize=9, color='#1f77b4')
    #     plt.text(x + 0.8, y - 3.5, f'({x:.1f}, {y:.1f})', fontsize=8, color='#1f77b4')

    # # Plot left-wing points
    # plt.scatter(topic_df['left_precision'], topic_df['left_recall'], color='#d62728', label='Left', s=80, alpha=0.8)
    # for i in range(len(topic_df)):
    #     x = topic_df['left_precision'][i]
    #     y = topic_df['left_recall'][i]
    #     plt.text(x + 0.8, y + 0.8, topic_df['topic'][i], fontsize=9, color='#d62728')
    #     plt.text(x + 0.8, y - 3.5, f'({x:.1f}, {y:.1f})', fontsize=8, color='#d62728')

    # plt.xlabel('Precision (%)')
    # plt.ylabel('Recall (%)')
    # plt.title('Precision vs Recall by Topic (Left vs Right)')
    # plt.xlim(60, 100)
    # plt.ylim(50, 100)
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.4)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, 'topic_metrics_comparison.png'))
    # plt.close()

def visualize_confidence_levels(df, output_folder):
    """Analyze performance across confidence levels"""
    if 'confidence' not in df.columns or df['confidence'].isna().all():
        print("No confidence level information available")
        return
    
    # 1. Confidence level accuracy comparison
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
    
    # 2. Grouped by model and confidence level
    model_confidence = df.groupby(['model', 'confidence'])['accuracy'].mean().reset_index()
    
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x='model', y='accuracy', hue='confidence', data=model_confidence)
    
    # Add labels for grouped bars
    for i, confidence in enumerate(model_confidence['confidence'].unique()):
        for j, model in enumerate(model_confidence['model'].unique()):
            subset = model_confidence[(model_confidence['model']==model) & 
                                     (model_confidence['confidence']==confidence)]
            if not subset.empty:
                value = subset['accuracy'].values[0]
                x = j + (i - 0.5) * 0.4  # Adjust x position to accommodate different grouped bars
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
    """Create topic-confidence heatmap"""
    if 'topic' not in df.columns or 'confidence' not in df.columns:
        print("No topic or confidence information available")
        return
    
    # Prepare heatmap data
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize political stance analysis results')
    parser.add_argument('folder_path', help='Folder containing analysis files')
    parser.add_argument('--output', help='Output folder for visualizations')
    
    args = parser.parse_args()
    
    analyze_and_visualize_political_data(args.folder_path, args.output)