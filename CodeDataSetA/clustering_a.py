import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ClusterAnalyzer:
    def __init__(self):
        # Define o caminho para a pasta results_clustering
        self.results_path = 'CodeDataSetA/results_clustering'
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
    
    def perform_clustering(self, X, y):
        """Realiza análise de clustering com K-means"""
        print("\n=== Iniciando Análise de Clustering ===")
        
        # Número de clusters igual ao número de classes
        n_clusters = len(np.unique(y))
        print(f"\nNúmero de clusters: {n_clusters}")
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # PCA para visualização
        pca = PCA(n_components=3)  # Usando 3 componentes para visualização
        X_pca = pca.fit_transform(X)
        
        # Visualização 2D
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
        plt.title('Clusters via K-means (2 Primeiros Componentes)')
        plt.xlabel('Primeiro Componente Principal')
        plt.ylabel('Segundo Componente Principal')
        plt.colorbar(scatter, label='Cluster')
        plt.savefig(f'{self.results_path}/clusters_2d.png')
        plt.close()
        
        # Visualização 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                           c=cluster_labels, cmap='viridis')
        ax.set_title('Clusters via K-means (3 Componentes)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.colorbar(scatter, label='Cluster')
        plt.savefig(f'{self.results_path}/clusters_3d.png')
        plt.close()
        
        return cluster_labels, kmeans, pca

    def analyze_clusters(self, X, cluster_labels, kmeans, feature_names):
        """Analisa características dos clusters"""
        print("\n=== Análise das Características dos Clusters ===")
        
        # Criar DataFrame com features e labels
        df_cluster = pd.DataFrame(X, columns=feature_names)
        df_cluster['Cluster'] = cluster_labels
        
        # Calcular médias por cluster
        cluster_means = df_cluster.groupby('Cluster').mean()
        print("\nMédia das features por cluster:")
        print(cluster_means)
        
        # Visualizar médias das features mais importantes
        plt.figure(figsize=(15, 6))
        sns.heatmap(cluster_means, annot=True, cmap='coolwarm', center=0)
        plt.title('Médias das Features por Cluster')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/cluster_means_heatmap.png')
        plt.close()
        
        # Análise dos centróides
        print("\nCentróides dos clusters:")
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)
        print(centroids)
        
        return cluster_means

    def evaluate_clustering(self, y_true, cluster_labels):
        """Avalia a qualidade do clustering"""
        print("\n=== Avaliação do Clustering ===")
        
        # Adjusted Rand Index
        ari = adjusted_rand_score(y_true, cluster_labels)
        print(f"\nAdjusted Rand Index: {ari:.4f}")
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, cluster_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusão: Classes Reais vs Clusters')
        plt.xlabel('Clusters Previstos')
        plt.ylabel('Classes Reais')
        plt.savefig(f'{self.results_path}/clustering_confusion_matrix.png')
        plt.close()
        
        # Calcular pureza
        purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)
        print(f"Pureza do clustering: {purity:.4f}")
        
        return {
            'ARI': ari,
            'Purity': purity,
            'Confusion Matrix': cm
        }

if __name__ == "__main__":
    # Carregar dados pré-processados
    from preprocessamento_a import DataPreprocessor
    
    # Preparar dados
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_analyze('data/taiwanese_bankruptcy_A.csv')  # caminho relativo à pasta atual
    df_cleaned = preprocessor.clean_data(df)
    scaled_data, y = preprocessor.scale_data(df_cleaned)
    
    # Usar dados normalizados
    X = scaled_data['normalizer']
    feature_names = df_cleaned.drop('Bankrupt?', axis=1).columns
    
    # Realizar análise de clustering
    analyzer = ClusterAnalyzer()
    
    # Clustering e visualização
    cluster_labels, kmeans, pca = analyzer.perform_clustering(X, y)
    
    # Análise das características
    cluster_means = analyzer.analyze_clusters(X, cluster_labels, kmeans, feature_names)
    
    # Avaliação do clustering
    evaluation_results = analyzer.evaluate_clustering(y, cluster_labels)