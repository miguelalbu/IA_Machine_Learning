from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import os
import sys

# Corrigir o caminho para a pasta raiz do projeto
sys.path.append(r'C:\Users\migue\OneDrive\Área de Trabalho\IA_Machine_Learning')

from utils.visualization_metrics import MetricsVisualizer
from utils.cross_validation import CrossValidation

class ModelTrainer:
    def __init__(self):
        # Define o caminho para a pasta results_modelagem
        self.results_path = 'CodeDataSetA/results_modelagem'
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        # Definindo os modelos e seus hiperparâmetros
        self.models = {
            'Regressão Logística': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'max_iter': [1000]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            },
            'Árvore de Decisão': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {}
            },
            'Rede Neural': {
                'model': MLPClassifier(random_state=42),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'max_iter': [1000]
                }
            }
        }


    def train_and_evaluate(self, X, y):
        cv = CrossValidation(n_splits=5)

        results = {}
        for name, model_info in self.models.items():
            print(f"\nAvaliando modelo: {name}")
            # Usar o modelo base, não o dicionário inteiro
            model = model_info['model']
            results[name] = cv.evaluate_model(model, X, y)

        return results

    def print_results(self, results):
        """Imprime os resultados de forma organizada"""
        print("\n=== Resultados Finais ===")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, values in metrics.items():
                print(f"{metric_name}:")
                print(f"  Média: {values['Média']:.4f}")
                print(f"  Desvio Padrão: {values['Desvio Padrão']:.4f}")

# Exemplo de uso
if __name__ == "__main__":
    # Carregar dados pré-processados (assumindo que já foram processados)
    # Substitua isso pelos seus dados reais
    from preprocessamento_a import DataPreprocessor

    preprocessor = DataPreprocessor()

    # Ajustar o caminho do arquivo
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset_limpo.csv')
    df = preprocessor.load_and_analyze(data_path)
    df_cleaned = preprocessor.clean_data(df)
    scaled_data, y = preprocessor.scale_data(df_cleaned)

    # Usar dados normalizados para treinamento
    X = scaled_data['normalizer']

    # Treinar e avaliar modelos
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X, y)
    trainer.print_results(results)
    visualizer = MetricsVisualizer('results/metrics')
    visualizer.plot_classifier_metrics(results, 'A')