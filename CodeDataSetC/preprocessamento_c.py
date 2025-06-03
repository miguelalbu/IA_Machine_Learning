import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataPreprocessorC:
    def __init__(self):
        self.results_path = 'results'
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
            
    def load_and_analyze(self, filepath):
        """Carrega e analisa os dados de ocupação"""
        print("=== Carregando e Analisando Dados ===\n")
        
        # Carregar dados
        df = pd.read_csv(filepath)
        
        # Análise inicial
        print("Dimensões do dataset:", df.shape)
        print("\nTipos de dados:")
        print(df.dtypes)
        
        # Estatísticas descritivas da variável alvo
        target = 'Room_Occupancy_Count'
        print("\nEstatísticas da variável alvo:")
        print(df[target].describe())
        
        return df
    
    def clean_data(self, df):
        """Limpa e prepara os dados"""
        print("\n=== Limpando Dados ===\n")
        
        # Verificar valores ausentes
        missing = df.isnull().sum()
        if missing.any():
            print("Valores ausentes:")
            print(missing[missing > 0])
            # Imputar pela mediana se necessário
            df = df.fillna(df.median())
            
        # Remover colunas não numéricas (se houver)
        df = df.select_dtypes(include=[np.number])
        
        return df
    
    def scale_data(self, df):
        """Aplica diferentes técnicas de escalonamento"""
        print("\n=== Aplicando Escalonamento ===\n")
        
        # Separar features e target
        X = df.drop('Room_Occupancy_Count', axis=1)
        y = df['Room_Occupancy_Count']
        
        # Aplicar diferentes escaladores
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        scaled_data = {}
        for name, scaler in scalers.items():
            scaled_data[name] = scaler.fit_transform(X)
            print(f"\nEstatísticas após {name}:")
            print(f"Média: {scaled_data[name].mean():.4f}")
            print(f"Desvio Padrão: {scaled_data[name].std():.4f}")
        
        return scaled_data, X, y
    
    def create_exploratory_analysis(self, df):
        """Cria visualizações para análise exploratória"""
        output_dir = os.path.join(self.results_path, 'exploratory')
        os.makedirs(output_dir, exist_ok=True)
        
        # Separar features numéricas
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 1. Distribuição da variável alvo
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Room_Occupancy_Count')
        plt.title('Distribuição da Ocupação')
        plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
        plt.close()
        
        # 2. Matriz de correlação (apenas variáveis numéricas)
        plt.figure(figsize=(15, 12))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Matriz de Correlação (Features Numéricas)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()
        
        # 3. Boxplots das features numéricas
        plt.figure(figsize=(15, 6))
        numeric_df.drop('Room_Occupancy_Count', axis=1).boxplot()
        plt.title('Distribuição das Features Numéricas')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'features_distribution.png'))
        plt.close()
        
        # 4. Análise temporal (opcional)
        plt.figure(figsize=(15, 6))
        df['Date'] = pd.to_datetime(df['Date'])
        df.groupby('Date')['Room_Occupancy_Count'].mean().plot()
        plt.title('Média de Ocupação por Data')
        plt.xlabel('Data')
        plt.ylabel('Ocupação Média')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_analysis.png'))
        plt.close()
        
    def get_target_statistics(self, df):
        """Calcula estatísticas detalhadas da variável alvo"""
        target = df['Room_Occupancy_Count']
        stats = {
            'Média': target.mean(),
            'Mediana': target.median(),
            'Desvio Padrão': target.std(),
            'Variância': target.var(),
            'Mínimo': target.min(),
            'Máximo': target.max(),
            'Quartis': target.quantile([0.25, 0.5, 0.75]).to_dict()
        }
        
        print("\n=== Estatísticas Detalhadas da Variável Alvo ===")
        for metric, value in stats.items():
            if metric != 'Quartis':
                print(f"{metric}: {value:.4f}")
            else:
                print("\nQuartis:")
                for q, v in value.items():
                    print(f"{q*100}%: {v:.4f}")
        
        return stats