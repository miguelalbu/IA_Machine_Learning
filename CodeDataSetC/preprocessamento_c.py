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