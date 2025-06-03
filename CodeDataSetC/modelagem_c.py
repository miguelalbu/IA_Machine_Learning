import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

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

if __name__ == "__main__":
    from preprocessamento_c import DataPreprocessorC
    
    # Preparar dados
    preprocessor = DataPreprocessorC()
    df = preprocessor.load_and_analyze('data/Occupancy_Estimation_C.csv')
    df_cleaned = preprocessor.clean_data(df)
    scaled_data, X, y = preprocessor.scale_data(df_cleaned)
    
    # Treinar e avaliar modelos
    trainer = RegressionTrainer()
    results = trainer.train_and_evaluate(scaled_data['standard'], y)
    trainer.plot_results(results)