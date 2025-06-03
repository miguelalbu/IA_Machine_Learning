import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Dados simulados das métricas do Dataset B (com foco no desbalanceamento extremo)
results = {
    'Regressão Logística': {
        'Balanced Accuracy': {'Média': 0.856, 'Desvio Padrão': 0.023},
        'Precisão': {'Média': 0.712, 'Desvio Padrão': 0.045},
        'Recall': {'Média': 0.714, 'Desvio Padrão': 0.038},
        'F1': {'Média': 0.713, 'Desvio Padrão': 0.041}
    },
    'KNN': {
        'Balanced Accuracy': {'Média': 0.723, 'Desvio Padrão': 0.035},
        'Precisão': {'Média': 0.685, 'Desvio Padrão': 0.042},
        'Recall': {'Média': 0.447, 'Desvio Padrão': 0.056},
        'F1': {'Média': 0.541, 'Desvio Padrão': 0.048}
    },
    'Árvore de Decisão': {
        'Balanced Accuracy': {'Média': 0.701, 'Desvio Padrão': 0.089},
        'Precisão': {'Média': 0.623, 'Desvio Padrão': 0.112},
        'Recall': {'Média': 0.403, 'Desvio Padrão': 0.098},
        'F1': {'Média': 0.489, 'Desvio Padrão': 0.104}
    },
    'Naive Bayes': {
        'Balanced Accuracy': {'Média': 0.734, 'Desvio Padrão': 0.028},
        'Precisão': {'Média': 0.667, 'Desvio Padrão': 0.035},
        'Recall': {'Média': 0.468, 'Desvio Padrão': 0.042},
        'F1': {'Média': 0.550, 'Desvio Padrão': 0.038}
    }
}

def plot_metrics_comparison_b(results, output_path='results_metrics_b'):
    os.makedirs(output_path, exist_ok=True)
    
    # Criar lista para armazenar os dados
    data_list = []
    
    # Preparar dados para visualização
    for model_name, metrics in results.items():
        for metric_name, values in metrics.items():
            data_list.append({
                'Modelo': model_name,
                'Métrica': metric_name,
                'Valor': values['Média']
            })
    
    # Criar DataFrame
    metrics_df = pd.DataFrame(data_list)
    
    # 1. Gráfico de barras com ênfase no balanced accuracy
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df, x='Modelo', y='Valor', hue='Métrica')
    plt.title('Comparação de Métricas por Modelo - Dataset B (MiniBooNE)')
    plt.xlabel('Modelos')
    plt.ylabel('Valor da Métrica')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'metrics_comparison_B.png'))
    plt.close()
    
    # 2. Heatmap com anotações
    plt.figure(figsize=(10, 8))
    metrics_pivot = metrics_df.pivot(index='Modelo', columns='Métrica', values='Valor')
    sns.heatmap(metrics_pivot, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Heatmap das Métricas - Dataset B (MiniBooNE)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'metrics_heatmap_B.png'))
    plt.close()

if __name__ == "__main__":
    plot_metrics_comparison_b(results)