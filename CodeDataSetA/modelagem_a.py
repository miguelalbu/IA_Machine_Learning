import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        """Treina e avalia todos os modelos usando validação cruzada"""
        results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model_info in self.models.items():
            print(f"\nTreinando {name}...")
            model = model_info['model']
            params = model_info['params']
            
            # GridSearchCV
            if params:
                grid_search = GridSearchCV(
                    model, params, cv=skf, scoring='f1',
                    n_jobs=-1, verbose=1
                )
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_
                print(f"Melhores parâmetros: {grid_search.best_params_}")
            else:
                best_model = model.fit(X, y)
            
            # Validação cruzada com o melhor modelo
            fold_results = []
            prob_correct = []
            prob_incorrect = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_val)
                y_prob = best_model.predict_proba(X_val)[:, 1]
                
                # Coletar probabilidades para análise
                correct_mask = y_pred == y_val
                prob_correct.extend(y_prob[correct_mask])
                prob_incorrect.extend(y_prob[~correct_mask])
                
                # Calcular métricas
                metrics = {
                    'Acurácia': accuracy_score(y_val, y_pred),
                    'Precisão': precision_score(y_val, y_pred),
                    'Recall': recall_score(y_val, y_pred),
                    'F1': f1_score(y_val, y_pred),
                    'AUC-ROC': roc_auc_score(y_val, y_prob)
                }
                fold_results.append(metrics)
                
                # Plotar matriz de confusão para cada fold
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(y_val, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{name} - Matriz de Confusão (Fold {fold})')
                plt.ylabel('Real')
                plt.xlabel('Predito')
                plt.savefig(f'{self.results_path}/{name.replace(" ", "_")}_cm_fold_{fold}.png')
                plt.close()
                
                # Plotar curva ROC
                fpr, tpr, _ = roc_curve(y_val, y_prob)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'AUC = {metrics["AUC-ROC"]:.2f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('Taxa de Falso Positivo')
                plt.ylabel('Taxa de Verdadeiro Positivo')
                plt.title(f'{name} - Curva ROC (Fold {fold})')
                plt.legend()
                plt.savefig(f'{self.results_path}/{name.replace(" ", "_")}_roc_fold_{fold}.png')
                plt.close()
            
            # Análise das probabilidades
            plt.figure(figsize=(10, 6))
            plt.hist(prob_correct, alpha=0.5, label='Classificações Corretas', bins=20)
            plt.hist(prob_incorrect, alpha=0.5, label='Classificações Incorretas', bins=20)
            plt.title(f'{name} - Distribuição das Probabilidades')
            plt.xlabel('Probabilidade Predita')
            plt.ylabel('Frequência')
            plt.legend()
            plt.savefig(f'{self.results_path}/{name.replace(" ", "_")}_prob_dist.png')
            plt.close()
            
            # Calcular médias das métricas
            mean_metrics = {}
            for metric in fold_results[0].keys():
                values = [fold[metric] for fold in fold_results]
                mean_metrics[metric] = {
                    'Média': np.mean(values),
                    'Desvio Padrão': np.std(values)
                }
            
            results[name] = mean_metrics
            
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
    df = preprocessor.load_and_analyze('data/taiwanese_bankruptcy_A.csv')
    df_cleaned = preprocessor.clean_data(df)
    scaled_data, y = preprocessor.scale_data(df_cleaned)
    
    # Usar dados normalizados para treinamento
    X = scaled_data['normalizer']
    
    # Treinar e avaliar modelos
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X, y)
    trainer.print_results(results)