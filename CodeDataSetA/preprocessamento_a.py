import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

class DataPreprocessor:
    def __init__(self):
        # Define o caminho absoluto para a pasta results
        self.results_path = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    def load_and_analyze(self, filepath):
        """Carrega e realiza análise inicial dos dados"""
        print("=== Carregando e Analisando Dados ===")
        df = pd.read_csv(filepath)

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

    def select_important_features(self, df, correlation_threshold=0.3):
        """Seleciona features mais importantes baseado na correlação com 'bankrupt?'"""
        print("\n=== Selecionando Features Importantes ===")

        # Calcular correlações com a variável target
        correlations = df.corr()['bankrupt?'].abs()

        # Selecionar features com correlação acima do threshold
        important_features = correlations[correlations > correlation_threshold].index

        # Manter sempre a variável target
        if 'bankrupt?' not in important_features:
            important_features = important_features.append(pd.Index(['bankrupt?']))

        print(f"Número de features selecionadas: {len(important_features)}")
        print("Features mantidas:")
        print(important_features.tolist())

        return df[important_features]

    def visualize_distributions(self, df):
        """Cria visualizações das distribuições"""
        print("\n=== Criando Visualizações ===")

        # Histogramas
        df.hist(figsize=(15, 10))
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/histograms.png')
        plt.close()

        # Boxplots para detectar outliers
        plt.figure(figsize=(15, 10))
        df.boxplot()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/boxplots.png')
        plt.close()

        # Matriz de correlação melhorada
        plt.figure(figsize=(12, 10))  # Tamanho ajustado

        # Calcular correlação
        corr_matrix = df.corr()

        # Criar máscara para triângulo superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Criar heatmap com melhorias
        sns.heatmap(corr_matrix,
                    mask=mask,
                    annot=True,  # Mostrar valores
                    fmt='.2f',   # Formato dos números (2 casas decimais)
                    cmap='RdBu',  # Mudança no esquema de cores
                    square=True,
                    linewidths=1,
                    annot_kws={'size': 6},  # Tamanho menor dos números
                    cbar_kws={'shrink': .5})  # Barra de cor menor

        plt.title('Matriz de Correlação', pad=20, size=14)

        # Rotacionar e ajustar labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Ajustar layout
        plt.tight_layout()

        # Salvar com maior resolução
        plt.savefig(f'{self.results_path}/correlation_matrix.png',
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white')
        plt.close()

    def clean_data(self, df):
        """Limpa e prepara os dados"""
        print("\n=== Limpando Dados ===")

        # 1. Tratamento de valores ausentes
        for column in df.columns:
            if df[column].isnull().any():
                if df[column].dtype in ['int64', 'float64']:
                    median = df[column].median()
                    df[column].fillna(median, inplace=True)
                    print(f"Preenchido valores ausentes em {column} com mediana: {median}")
                else:
                    mode = df[column].mode()[0]
                    df[column].fillna(mode, inplace=True)
                    print(f"Preenchido valores ausentes em {column} com moda: {mode}")

        # 2. Remoção de variáveis altamente correlacionadas
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

        if to_drop:
            print(f"\nRemovendo {len(to_drop)} variáveis altamente correlacionadas:")
            print(to_drop)
            df = df.drop(to_drop, axis=1)

        return df

    def scale_data(self, df):
        """Aplica diferentes técnicas de escalonamento"""
        print("\n=== Aplicando Escalonamento ===")

        # Separar features e target - usando o nome correto da coluna em minúsculas
        X = df.drop('bankrupt?', axis=1)
        y = df['bankrupt?']

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
            pca = PCA(n_components=0.95)  # Mantém 95% da variância
            X_pca = pca.fit_transform(X_scaled)

            print(f"\nResultados PCA com {name}:")
            print(f"Número de componentes: {X_pca.shape[1]}")
            print(f"Variância explicada: {np.sum(pca.explained_variance_ratio_):.4f}")

            pca_results[name] = {
                'X_pca': X_pca,
                'pca': pca
            }

            # Visualização 2D se possível
            if X_pca.shape[1] >= 2:
                plt.figure(figsize=(8, 6))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
                plt.title(f'PCA - Primeiros 2 Componentes ({name})')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.savefig(f'{self.results_path}/pca_scatter_{name}.png')
                plt.close()

        return pca_results

# Exemplo de uso
if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    # Usar caminho absoluto para o dataset
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset_limpo.csv')

    # Verificar se o arquivo existe
    if not os.path.exists(data_path):
        print(f"Erro: Arquivo não encontrado em: {data_path}")
        print("Verifique se o arquivo existe e se o caminho está correto.")
        sys.exit(1)

    # Carregar e analisar dados
    df = preprocessor.load_and_analyze(data_path)

    # Selecionar features importantes com threshold menor
    df_important = preprocessor.select_important_features(df, correlation_threshold=0.2)

    # Criar visualizações com dataset reduzido
    preprocessor.visualize_distributions(df_important)

    # Limpar dados
    df_cleaned = preprocessor.clean_data(df_important)

    # Escalar dados
    scaled_data, y = preprocessor.scale_data(df_cleaned)

    # Reduzir dimensionalidade
    pca_results = preprocessor.reduce_dimensions(scaled_data, y)

    print("\n=== Processamento Concluído ===")
    print("Verifique a pasta 'results' para visualizações geradas")