import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        # Define o caminho para a pasta results
        self.results_path = 'results'
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
            
    def load_and_analyze(self, filepath):
        """Carrega e realiza análise inicial dos dados"""
        print("=== Carregando e Analisando Dados ===")
        
        try:
            # Ler o arquivo CSV com o cabeçalho
            df = pd.read_csv(filepath)
            
            # Identificar a coluna target (última coluna)
            target_col = df.columns[-1]
            
            # Converter target para binário (0 e 1)
            df[target_col] = (df[target_col] > 0).astype(int)
            
            # Informações básicas
            print("\nDimensões do dataset:", df.shape)
            print("\nDistribuição das classes:")
            print(df[target_col].value_counts())
            print("\nTipos de dados:")
            print(df.dtypes)
            
            return df
            
        except Exception as e:
            print(f"Erro ao carregar o arquivo: {e}")
            raise

    def visualize_distributions(self, df):
        """Cria visualizações das distribuições"""
        print("\n=== Criando Visualizações ===")
        
        # Amostragem para visualização
        df_sample = df.sample(n=min(1000, len(df)), random_state=42)
        
        # Matriz de correlação melhorada
        plt.figure(figsize=(20, 16))
        
        # Calcular correlação
        corr_matrix = df_sample.corr()
        
        # Criar máscara para triângulo superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Criar heatmap com melhorias
        sns.heatmap(corr_matrix,
                    mask=mask,
                    annot=False,  # Remove anotações numéricas
                    fmt='.2f',
                    cmap='RdBu_r',  # Melhor esquema de cores
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'shrink': .8},
                    xticklabels=False,  # Remove labels dos eixos
                    yticklabels=False)
        
        plt.title('Matriz de Correlação - Dataset B\n(Amostra de 1000 registros)', 
                  pad=20, 
                  size=16)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar em alta resolução
        plt.savefig(f'{self.results_path}/correlation_matrix.png',
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white')
        plt.close()

# Código de execução
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Definir caminho do arquivo
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset_limpo.csv')
    
    # Verificar se arquivo existe
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
    
    # Executar processamento
    df = preprocessor.load_and_analyze(data_path)
    preprocessor.visualize_distributions(df)
