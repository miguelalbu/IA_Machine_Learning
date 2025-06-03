import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ClusterAnalyzer:
    def __init__(self):
        # Define o caminho para a pasta results_clustering
        self.results_path = 'results_clustering'
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
    
    def perform_clustering(self, X, y):
        """Realiza análise de clustering com K-means"""
        print("\n=== Iniciando Análise de Clustering ===")
        
        # Número fixo de clusters (2 ou 3 é mais apropriado para este dataset)
        n_clusters = 2  # MiniBooNE é um problema binário
        print(f"\nNúmero de clusters: {n_clusters}")
        
        # Aplicar K-means com amostragem para datasets grandes
        # Usar uma amostra para treinamento inicial
        sample_size = 10000  # ou outro tamanho apropriado
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
        
        # Treinar K-means
        kmeans = KMeans(n_clusters=n_clusters, 
                        random_state=42, 
                        n_init=10,
                        max_iter=300)
        
        print("Treinando K-means...")
        cluster_labels = kmeans.fit_predict(X)
        
        # PCA para visualização (usar amostra para PCA também)
        print("Aplicando PCA...")
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_sample)
        labels_sample = cluster_labels[indices]
        
        # Visualização 2D
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=labels_sample, 
                             cmap='viridis',
                             alpha=0.6)
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
                            c=labels_sample, 
                            cmap='viridis',
                            alpha=0.6)
        ax.set_title('Clusters via K-means (3 Componentes)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.colorbar(scatter, label='Cluster')
        plt.savefig(f'{self.results_path}/clusters_3d.png')
        plt.close()
        
        return cluster_labels, kmeans, pca

    def evaluate_clustering(self, y_true, cluster_labels):
        """Avalia a qualidade do clustering"""
        print("\n=== Avaliação do Clustering ===")
        
        # Converter y_true e cluster_labels para inteiros
        y_true = y_true.astype(int)
        cluster_labels = cluster_labels.astype(int)
        
        # Garantir que os clusters correspondam às classes corretas
        if np.mean(cluster_labels == 1) > np.mean(cluster_labels == 0):
            cluster_labels = 1 - cluster_labels  # Inverter labels se necessário
        
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
        
        # Adicionar mais informações sobre os clusters
        print("\nInformações dos Clusters:")
        for i in range(2):
            print(f"\nCluster {i}:")
            print(f"Número de amostras: {np.sum(cluster_labels == i)}")
            print(f"Proporção no dataset: {np.mean(cluster_labels == i):.4f}")
        
        return {
            'ARI': ari,
            'Purity': purity,
            'Confusion Matrix': cm
        }

if __name__ == "__main__":
    # Carregar dados pré-processados
    from preprocessamento_b import DataPreprocessor
    
    # Preparar dados
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_analyze('data/MiniBooNE_B.txt')
    df_cleaned = preprocessor.clean_data(df)
    scaled_data, y = preprocessor.scale_data(df_cleaned)
    
    # Usar dados normalizados
    X = scaled_data['normalizer']
    
    # Converter target para inteiros antes do clustering
    y = y.astype(int)
    
    # Realizar análise de clustering
    analyzer = ClusterAnalyzer()
    
    # Clustering e visualização
    cluster_labels, kmeans, pca = analyzer.perform_clustering(X, y)
    
    # Avaliação do clustering
    evaluation_results = analyzer.evaluate_clustering(y, cluster_labels)