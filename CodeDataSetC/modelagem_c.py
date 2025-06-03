import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from preprocessamento_c import DataPreprocessorC

class RegressionTrainer:
    def __init__(self):
        self.results_path = 'results'
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        
        # Definir modelos de regressão
        self.models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
        }
    
    def calculate_mape(self, y_true, y_pred):
        """Calcula o MAPE (Mean Absolute Percentage Error)"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def train_and_evaluate(self, X, y):
        """Treina e avalia os modelos de regressão"""
        results = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nTreinando {name}...")
            
            # Validação cruzada
            scores = cross_validate(model, X, y,
                                 scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                                 cv=kf)
            
            # Calcular métricas
            r2 = scores['test_r2'].mean()
            rmse = np.sqrt(-scores['test_neg_mean_squared_error'].mean())
            mae = -scores['test_neg_mean_absolute_error'].mean()
            
            results[name] = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae
            }
            
            print(f"R²: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
        
        return results
    
    def plot_results(self, results):
        """Plota comparação dos resultados"""
        metrics_df = pd.DataFrame(results).T
        
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar')
        plt.title('Comparação das Métricas por Modelo')
        plt.xlabel('Modelo')
        plt.ylabel('Valor')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/regression_comparison.png')
        plt.close()
    
    def compare_metrics_with_stats(self, results, target_stats):
        """Compara métricas do modelo com estatísticas da variável alvo"""
        print("\n=== Comparação com Estatísticas Base ===")
        
        for model_name, metrics in results.items():
            print(f"\nModelo: {model_name}")
            print(f"RMSE: {metrics['RMSE']:.4f} (vs Desvio Padrão: {target_stats['Desvio Padrão']:.4f})")
            print(f"MAE: {metrics['MAE']:.4f} (vs Média: {target_stats['Média']:.4f})")
            
            # Calcular erro relativo às estatísticas base
            rmse_ratio = metrics['RMSE'] / target_stats['Desvio Padrão']
            mae_ratio = metrics['MAE'] / target_stats['Média']
            
            print(f"RMSE/Desvio: {rmse_ratio:.4f}")
            print(f"MAE/Média: {mae_ratio:.4f}")
    
    def plot_metrics_comparison(self, results, target_stats):
        """Plota comparação entre métricas e estatísticas base"""
        plt.figure(figsize=(12, 6))
        
        models = list(results.keys())
        metrics = ['RMSE', 'MAE']
        
        x = np.arange(len(models))
        width = 0.35
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            plt.bar(x + i*width, values, width, label=metric)
        
        # Adicionar linhas de referência
        plt.axhline(y=target_stats['Desvio Padrão'], color='r', linestyle='--', label='Desvio Padrão Base')
        plt.axhline(y=target_stats['Média'], color='g', linestyle='--', label='Média Base')
        
        plt.xlabel('Modelos')
        plt.ylabel('Valor')
        plt.title('Comparação de Métricas vs Estatísticas Base')
        plt.xticks(x + width/2, models)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.results_path, 'metrics_vs_stats.png'))
        plt.close()
    
    def plot_all_metrics(self, results):
        """
        Cria visualizações agregadas comparando todas as métricas dos modelos
        """
        output_dir = os.path.join(self.results_path, 'metrics')
        os.makedirs(output_dir, exist_ok=True)
        
        # Organizar dados para plotagem
        metrics_df = pd.DataFrame(columns=['Modelo', 'Métrica', 'Valor'])
        
        for model_name, metrics in results.items():
            for metric, value in metrics.items():
                metrics_df = metrics_df.append({
                    'Modelo': model_name,
                    'Métrica': metric,
                    'Valor': value
                }, ignore_index=True)
        
        # 1. Gráfico de barras agrupadas
        plt.figure(figsize=(12, 6))
        sns.barplot(data=metrics_df, x='Modelo', y='Valor', hue='Métrica')
        plt.title('Comparação de Métricas por Modelo')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison_bar.png'))
        plt.close()
        
        # 2. Heatmap das métricas
        plt.figure(figsize=(10, 8))
        metrics_pivot = metrics_df.pivot(index='Modelo', columns='Métrica', values='Valor')
        sns.heatmap(metrics_pivot, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Heatmap das Métricas por Modelo')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
        plt.close()
        
        # 3. Gráfico de radar (para visualização multidimensional)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        models = metrics_pivot.index
        metrics = metrics_pivot.columns
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        
        for model in models:
            values = metrics_pivot.loc[model].values
            values = np.concatenate((values, [values[0]]))  # Fechar o polígono
            angles_plot = np.concatenate((angles, [angles[0]]))  # Fechar o polígono
            
            ax.plot(angles_plot, values, 'o-', linewidth=2, label=model)
            ax.fill(angles_plot, values, alpha=0.25)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics)
        plt.title('Gráfico de Radar - Métricas por Modelo')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_radar.png'))
        plt.close()

if __name__ == "__main__":
    # Carregar e preparar dados
    preprocessor = DataPreprocessorC()
    df = preprocessor.load_and_analyze('data/Occupancy_Estimation_C.csv')
    
    # Análise exploratória e estatísticas
    target_stats = preprocessor.get_target_statistics(df)
    preprocessor.create_exploratory_analysis(df)
    
    # Pré-processamento
    df_cleaned = preprocessor.clean_data(df)
    scaled_data, X, y = preprocessor.scale_data(df_cleaned)
    
    # Modelagem e avaliação
    trainer = RegressionTrainer()
    results = trainer.train_and_evaluate(scaled_data['standard'], y)
    
    # Gerar visualizações das métricas
    trainer.plot_all_metrics(results)