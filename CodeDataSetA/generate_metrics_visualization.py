import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Dados simulados das métricas do Dataset A (baseados em resultados típicos)
results = {
    'Regressão Logística': {
        'Acurácia': {'Média': 0.967, 'Desvio Padrão': 0.012},
        'Precisão': {'Média': 0.915, 'Desvio Padrão': 0.018},
        'Recall': {'Média': 0.662, 'Desvio Padrão': 0.029},
        'F1': {'Média': 0.767, 'Desvio Padrão': 0.025}
    },
    'KNN': {
        'Acurácia': {'Média': 0.966, 'Desvio Padrão': 0.013},
        'Precisão': {'Média': 0.662, 'Desvio Padrão': 0.029},
        'Recall': {'Média': 0.662, 'Desvio Padrão': 0.029},
        'F1': {'Média': 0.662, 'Desvio Padrão': 0.029}
    },
    'Árvore de Decisão': {
        'Acurácia': {'Média': 0.959, 'Desvio Padrão': 0.047},
        'Precisão': {'Média': 0.679, 'Desvio Padrão': 0.102},
        'Recall': {'Média': 0.679, 'Desvio Padrão': 0.102},
        'F1': {'Média': 0.679, 'Desvio Padrão': 0.102}
    },
    'Naive Bayes': {
        'Acurácia': {'Média': 0.967, 'Desvio Padrão': 0.012},
        'Precisão': {'Média': 0.915, 'Desvio Padrão': 0.018},
        'Recall': {'Média': 0.662, 'Desvio Padrão': 0.029},
        'F1': {'Média': 0.767, 'Desvio Padrão': 0.025}
    }
}

def plot_metrics_comparison(results, output_path='results_metrics'):
    os.makedirs(output_path, exist_ok=True)
    
    # Criar lista para armazenar os dados
    data_list = []
    
    # Preparar dados para visualização de forma otimizada
    for model_name, metrics in results.items():
        for metric_name, values in metrics.items():
            data_list.append({
                'Modelo': model_name,
                'Métrica': metric_name,
                'Valor': values['Média']
            })
    
    # Criar DataFrame de uma vez
    metrics_df = pd.DataFrame(data_list)
    
    # 1. Gráfico de barras
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df, x='Modelo', y='Valor', hue='Métrica')
    plt.title('Comparação de Métricas por Modelo - Dataset A')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'metrics_comparison_A.png'))
    plt.close()
    
    # 2. Heatmap
    plt.figure(figsize=(10, 8))
    metrics_pivot = metrics_df.pivot(index='Modelo', columns='Métrica', values='Valor')
    sns.heatmap(metrics_pivot, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Heatmap das Métricas - Dataset A')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'metrics_heatmap_A.png'))
    plt.close()

if __name__ == "__main__":
    plot_metrics_comparison(results)