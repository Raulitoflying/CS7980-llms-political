import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re

def analyze_and_visualize_matrix(folder_paths, output_folder=None):
    """Generate model cross-analysis matrix only"""
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]  # Convert to list if it's a single folder
    
    if output_folder is None:
        # Default output to matrix_output subfolder in the parent directory of the first folder
        parent_dir = os.path.dirname(os.path.abspath(folder_paths[0]))
        output_folder = os.path.join(parent_dir, 'matrix_output')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all analysis files
    analysis_files = []
    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('_analysis.json'):
                    analysis_files.append(os.path.join(root, file))
    
    print(f"Found {len(analysis_files)} analysis files across {len(folder_paths)} folders")
    
    # Extract file data
    file_data = []
    
    # Model name mapping
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
    
    # Determine which models are data generators (from folder names)
    generator_models = set()
    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path)
        
        # Extract model from folder name
        for key, value in model_name_mapping.items():
            if key.lower() in folder_name.lower():
                generator_models.add(value)
                break
    
    print(f"Identified generator models from folder names: {', '.join(generator_models)}")
    
    for file_path in analysis_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Basic file info
            file_name = os.path.basename(file_path)
            
            # Get analyzer model name from JSON data
            full_model_name = data.get('model_name', 'unknown')
            
            # Simplify analyzer model name
            analyzer_model = 'Unknown'
            for key, value in model_name_mapping.items():
                if key in full_model_name.lower():
                    analyzer_model = value
                    break
            
            # If still unknown, try to extract from filename
            if analyzer_model == 'Unknown':
                # Try to extract from filename format: balanced20Posts_[confidence]_[topic]_[generator]_[analyzer]_analysis.json
                parts = file_name.split('_')
                if len(parts) >= 5:  # Ensure filename has enough parts
                    # Analyzer is usually the part before the last underscore
                    for i in range(len(parts) - 2, 0, -1):  # Start from second-to-last part and search backwards
                        for key, value in model_name_mapping.items():
                            if key.lower() in parts[i].lower():
                                analyzer_model = value
                                break
                        if analyzer_model != 'Unknown':
                            break
            
            # Identify generator model from filename
            generator_model = 'Unknown'
            
            # Try to extract from filename format: balanced20Posts_[confidence]_[topic]_[generator]_[analyzer]_analysis.json
            match = re.search(r'(\w+)-(\w+)-(\w+)(?:-\w+)?_(\w+)_(\w+)', file_name)
            if match:
                generator_name = match.group(1)
                for key, value in model_name_mapping.items():
                    if key.lower() in generator_name.lower():
                        generator_model = value
                        break
            
            # If can't extract from filename, try using folder name
            if generator_model == 'Unknown':
                folder_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                for gen_model in generator_models:
                    if gen_model.lower() in folder_name.lower():
                        generator_model = gen_model
                        break
            
            # Extract metrics
            metrics = data.get('metrics', {})
            accuracy = metrics.get('overall_accuracy', 0)
            
            # Add to file data list
            file_entry = {
                'file': file_name,
                'analyzer_model': analyzer_model,
                'generator_model': generator_model,
                'accuracy': accuracy
            }
            
            file_data.append(file_entry)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create dataframe
    df = pd.DataFrame(file_data)
    
    if df.empty:
        print("No valid data found in the specified folders")
        return
    
    # Filter out unrecognized models
    df = df[(df['analyzer_model'] != 'Unknown') & (df['generator_model'] != 'Unknown')]
    
    if df.empty:
        print("No valid data after filtering unknown models")
        return
    
    # If generator_models is empty, extract all unique generator models from data
    if not generator_models:
        generator_models = set(df['generator_model'].unique())
    
    print(f"Analyzer models found: {', '.join(df['analyzer_model'].unique())}")
    print(f"Generator models to include in matrix: {', '.join(generator_models)}")
    
    # Create model cross-analysis matrix
    create_model_cross_analysis_matrix(df, output_folder, generator_models)
    
    print(f"Matrix visualizations created in {output_folder}")

def create_model_cross_analysis_matrix(df, output_folder, generator_models):
    """Create model cross-analysis matrix visualization"""
    # Ensure data has both analyzer and generator model information
    df_cross = df.dropna(subset=['analyzer_model', 'generator_model'])
    if df_cross.empty:
        print("No data with both analyzer and generator model information available")
        return
    
    # Calculate average accuracy for each analyzer model on each generator model
    cross_accuracy = df_cross.groupby(['analyzer_model', 'generator_model'])['accuracy'].mean().reset_index()
    
    # Create cross matrix
    try:
        # Pivot table
        pivot_data = cross_accuracy.pivot_table(
            values='accuracy', 
            index='analyzer_model', 
            columns='generator_model',
            aggfunc='mean'
        )
        
        # Ensure all specified generator models are included
        for gen_model in generator_models:
            if gen_model not in pivot_data.columns:
                pivot_data[gen_model] = np.nan
        
        # Keep only specified generator model columns
        pivot_data = pivot_data[[col for col in pivot_data.columns if col in generator_models]]
        
        # Draw heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100, 
                    linewidths=0.5, mask=pivot_data.isna())
        plt.title('Model Cross-Analysis Accuracy Matrix\n(Rows: Analyzer Models, Columns: Generator Models)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'model_cross_analysis_matrix.png'))
        plt.close()
        
        # Save data as CSV for further analysis
        pivot_data.to_csv(os.path.join(output_folder, 'model_cross_analysis_matrix.csv'))
        
        # Calculate relative performance of analyzer models vs. generator models
        # i.e., how each analyzer's accuracy compares to the average accuracy for each generator
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
        
        # Draw relative performance matrix
        try:
            pivot_rel = rel_perf_df.pivot_table(
                values='relative_accuracy', 
                index='analyzer_model', 
                columns='generator_model',
                aggfunc='mean'
            )
            
            # Ensure all specified generator models are included
            for gen_model in generator_models:
                if gen_model not in pivot_rel.columns:
                    pivot_rel[gen_model] = np.nan
                    
            pivot_rel = pivot_rel[[col for col in pivot_rel.columns if col in generator_models]]
            
            # Use diverging color palette
            plt.figure(figsize=(12, 10))
            sns.heatmap(pivot_rel, annot=True, fmt='.1f', cmap='RdBu_r', 
                        center=0, vmin=-20, vmax=20, linewidths=0.5,
                        mask=pivot_rel.isna())
            plt.title('Relative Performance Matrix\n(How much better/worse each analyzer performs compared to the average for each generator)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'model_relative_performance_matrix.png'))
            plt.close()
            
            # Save relative performance data
            pivot_rel.to_csv(os.path.join(output_folder, 'model_relative_performance_matrix.csv'))
            
        except Exception as e:
            print(f"Could not create relative performance matrix: {e}")
        
        # Create analyzer performance comparison chart
        try:
            analyzer_avg = cross_accuracy[cross_accuracy['generator_model'].isin(generator_models)]
            analyzer_avg = analyzer_avg.groupby('analyzer_model')['accuracy'].mean().reset_index()
            
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x='analyzer_model', y='accuracy', data=analyzer_avg)
            
            # Add value labels
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
            
            # Save analyzer average performance data
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