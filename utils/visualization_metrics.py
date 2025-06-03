import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class MetricsVisualizer:
    def __init__(self, output_path):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
    
    def plot_classifier_metrics(self, results, dataset_name):
        metrics_df = pd.DataFrame(columns=['Modelo', 'Métrica', 'Valor'])
        
        for model_name, metrics in results.items():
            for metric_name, values in metrics.items():
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Modelo': [model_name],
                    'Métrica': [metric_name],
                    'Valor': [values['Média']]
                })], ignore_index=True)
        
        # 1. Gráfico de barras
        plt.figure(figsize=(12, 6))
        sns.barplot(data=metrics_df, x='Modelo', y='Valor', hue='Métrica')
        plt.title(f'Métricas por Modelo - Dataset {dataset_name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'metrics_comparison_{dataset_name}.png'))
        plt.close()
        
        # 2. Heatmap
        plt.figure(figsize=(10, 8))
        metrics_pivot = metrics_df.pivot(index='Modelo', columns='Métrica', values='Valor')
        sns.heatmap(metrics_pivot, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title(f'Heatmap das Métricas - Dataset {dataset_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, f'metrics_heatmap_{dataset_name}.png'))
        plt.close()