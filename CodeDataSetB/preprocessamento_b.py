import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        # Define o caminho para a pasta results dentro de CodeDataSetB
        self.results_path = 'results'
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
            
    def load_and_analyze(self, filepath):
        """Carrega e realiza análise inicial dos dados"""
        print("=== Carregando e Analisando Dados ===")
        
        try:
            # Ler o arquivo linha por linha
            with open(filepath, 'r') as file:
                # Pular a primeira linha
                next(file)
                # Ler o resto do arquivo como matriz numpy
                data = np.loadtxt(file)
        
            # Converter para DataFrame
            feature_names = [f'feature_{i}' for i in range(data.shape[1]-1)]
            df = pd.DataFrame(data[:, :-1], columns=feature_names)
            df['target'] = data[:, -1]
            
            # Informações básicas
            print("\nDimensões do dataset:", df.shape)
            print("\nTipos de dados:")
            print(df.dtypes)
            
            # Valores ausentes
            missing = df.isnull().sum()
            if missing.any():
                print("\nValores ausentes encontrados:")
                print(missing[missing > 0])
            
            return df
            
        except Exception as e:
            print(f"Erro ao carregar o arquivo: {e}")
            raise  # Re-lança a exceção para debug
    
    def visualize_distributions(self, df):
        """Cria visualizações das distribuições"""
        print("\n=== Criando Visualizações ===")
        
        # Amostragem para visualização (dataset é muito grande)
        df_sample = df.sample(n=min(1000, len(df)), random_state=42)
        
        # Histogramas
        df_sample.hist(figsize=(20, 15), bins=50)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/histograms.png')
        plt.close()
        
        # Boxplots - usando lista de colunas ao invés de tupla
        feature_cols = [col for col in df_sample.columns if col != 'target']
        plt.figure(figsize=(20, 10))
        df_sample[feature_cols].boxplot()
        plt.xticks(rotation=90)
        plt.title('Distribuição das Features')
        plt.ylabel('Valores')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/boxplots.png')
        plt.close()
        
        # Matriz de correlação (usando amostra)
        plt.figure(figsize=(15, 15))
        sns.heatmap(df_sample.corr(), annot=False, cmap='coolwarm')
        plt.title('Matriz de Correlação')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/correlation_matrix.png')
        plt.close()
    
    def clean_data(self, df):
        """Limpa e prepara os dados"""
        print("\n=== Limpando Dados ===")
        
        # 1. Tratamento de valores ausentes
        for column in df.columns:
            if df[column].isnull().any():
                median = df[column].median()
                df[column].fillna(median, inplace=True)
                print(f"Preenchido valores ausentes em {column} com mediana: {median}")
        
        # 2. Remoção de variáveis altamente correlacionadas
        # Usando amostra para calcular correlações (dataset muito grande)
        df_sample = df.sample(n=min(1000, len(df)), random_state=42)
        corr_matrix = df_sample.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Não incluir a coluna 'target' na lista de colunas para remover
        to_drop = [column for column in upper.columns if column != 'target' and any(upper[column] > 0.95)]
        
        if to_drop:
            print(f"\nRemovendo {len(to_drop)} variáveis altamente correlacionadas:")
            print(to_drop)
            df = df.drop(to_drop, axis=1)
        
        return df
    
    def scale_data(self, df):
        """Aplica diferentes técnicas de escalonamento"""
        print("\n=== Aplicando Escalonamento ===")
        
        # Verificar se a coluna target existe antes de tentar removê-la
        if 'target' in df.columns:
            X = df.drop('target', axis=1)
            y = df['target']
        else:
            # Se não existir, assumir que a última coluna é o target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
    
        # Aplicar diferentes escaladores
        scalers = {
            'normalizer': MinMaxScaler(),
            'standardizer': StandardScaler(),
            'robust': RobustScaler()
        }
        
        scaled_data = {}
        for name, scaler in scalers.items():
            X_scaled = scaler.fit_transform(X)
            scaled_data[name] = X_scaled
            print(f"\nEstatísticas após {name}:")
            print(f"Média: {X_scaled.mean():.4f}")
            print(f"Desvio Padrão: {X_scaled.std():.4f}")
        
        return scaled_data, y
    
    def reduce_dimensions(self, scaled_data, y):
        """Aplica PCA para redução de dimensionalidade"""
        print("\n=== Reduzindo Dimensionalidade ===")
        
        pca_results = {}
        for name, X_scaled in scaled_data.items():
            pca = PCA(n_components=0.95)
            X_pca = pca.fit_transform(X_scaled)
            
            print(f"\nResultados PCA com {name}:")
            print(f"Número de componentes: {X_pca.shape[1]}")
            print(f"Variância explicada: {np.sum(pca.explained_variance_ratio_):.4f}")
            
            pca_results[name] = {
                'X_pca': X_pca,
                'pca': pca
            }
            
            if X_pca.shape[1] >= 2:
                plt.figure(figsize=(8, 6))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
                plt.title(f'PCA - Primeiros 2 Componentes ({name})')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.savefig(f'{self.results_path}/pca_scatter_{name}.png')
                plt.close()
        
        return pca_results

def create_sample_data():
    """Cria dados de exemplo para teste"""
    # Criar dados sintéticos baseados nas características do MiniBooNE
    n_samples = 130064
    n_features = 50
    
    # Gerar features aleatórias
    X = np.random.randn(n_samples, n_features)
    
    # Gerar target com desbalanceamento similar
    y = np.zeros(n_samples)
    y[:7] = 1  # 7 amostras positivas
    
    return X, y

def preprocess_data():
    # Configurar diretórios
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data')
    os.makedirs(data_path, exist_ok=True)
    
    try:
        # Tentar carregar dados originais
        print("Tentando carregar MiniBooNE_PID.txt...")
        df = pd.read_csv('MiniBooNE_PID.txt', delim_whitespace=True, header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    except FileNotFoundError:
        print("Arquivo original não encontrado, criando dados de exemplo...")
        X, y = create_sample_data()
    
    # Aplicar StandardScaler
    print("Aplicando normalização...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Salvar dados processados
    print("Salvando dados processados...")
    np.save(os.path.join(data_path, 'X_processed.npy'), X_scaled)
    np.save(os.path.join(data_path, 'y_processed.npy'), y)
    
    print(f"Dados salvos em: {data_path}")
    print(f"Shape X: {X_scaled.shape}")
    print(f"Shape y: {y.shape}")
    print(f"Classes únicas: {np.unique(y, return_counts=True)}")

if __name__ == "__main__":
    preprocess_data()
    preprocessor = DataPreprocessor()
    
    
    # Carregar e analisar dados - usando caminho relativo
    df = preprocessor.load_and_analyze('data/dataset_limpo.csv')
    
    # Criar visualizações
    preprocessor.visualize_distributions(df)
    
    # Limpar dados
    df_cleaned = preprocessor.clean_data(df)
    
    # Escalar dados
    scaled_data, y = preprocessor.scale_data(df_cleaned)
    
    # Reduzir dimensionalidade
    pca_results = preprocessor.reduce_dimensions(scaled_data, y)
    
    print("\n=== Processamento Concluído ===")