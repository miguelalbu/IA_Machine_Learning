import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, roc_curve, balanced_accuracy_score, make_scorer)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self):
        self.results_path = 'results_modelagem'
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        
        # Definindo modelos com parâmetros específicos para o dataset B
        self.models = {
            'Regressão Logística': {
                'model': LogisticRegression(max_iter=1000),
                'params': {'C': [0.1, 1.0, 10.0]}
            },
            'Random Forest': {
                'model': RandomForestClassifier(),
                'params': {'n_estimators': [100, 200]}
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {'n_neighbors': [3, 5, 7]}
            },
            'Árvore de Decisão': {
                'model': DecisionTreeClassifier(),
                'params': {'max_depth': [5, 10, None]}
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {}
            }
        }

    def train_and_evaluate(self, X, y, sample_weight=None):
        """Treina e avalia todos os modelos usando validação cruzada com balanceamento"""
        results = {}
        
        # Reduzir número de folds para 2 devido ao número muito pequeno de amostras positivas
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        
        
        smote = SMOTE(
            random_state=42,
            k_neighbors=1,  # Usar 1 vizinho devido ao número muito pequeno de amostras positivas
            sampling_strategy=0.1  # Gerar 10% do número de amostras majoritárias
        )
        
        print("\nAplicando SMOTE modificado para dados extremamente desbalanceados...")
        print("Distribuição original das classes:")
        print(f"Classe 0: {np.sum(y == 0)} amostras")
        print(f"Classe 1: {np.sum(y == 1)} amostras")
        
        for name, model_info in self.models.items():
            print(f"\nTreinando {name}...")
            model = model_info['model']
            params = model_info['params']
            
            # Criar pipeline do imbalanced-learn
            pipeline = ImbPipeline([
                ('smote', smote),
                ('classifier', model)
            ])
            
            # Adaptar parâmetros para pipeline
            pipeline_params = {'classifier__' + key: value for key, value in params.items()}
            
            # GridSearchCV com foco em métricas para classes desbalanceadas
            grid_search = GridSearchCV(
                pipeline, 
                pipeline_params,
                cv=skf,
                scoring={
                    'precision': make_scorer(precision_score, zero_division=0),
                    'recall': make_scorer(recall_score, zero_division=0),
                    'f1': make_scorer(f1_score, zero_division=0),
                    'balanced_accuracy': make_scorer(balanced_accuracy_score)
                },
                refit='balanced_accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            try:
                # Treinamento com tratamento de erro
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_
                print(f"Melhores parâmetros: {grid_search.best_params_}")
                
                # Validação cruzada com o melhor modelo
                fold_results = []
                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    try:
                        # Tentar balancear e treinar
                        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                        best_model.named_steps['classifier'].fit(X_train_balanced, y_train_balanced)
                        
                        # Predizer
                        y_pred = best_model.named_steps['classifier'].predict(X_val)
                        y_prob = best_model.named_steps['classifier'].predict_proba(X_val)[:, 1]
                        
                        # Calcular métricas focadas em desbalanceamento
                        metrics = {
                            'Balanced Accuracy': balanced_accuracy_score(y_val, y_pred),
                            'Precisão': precision_score(y_val, y_pred, zero_division=0),
                            'Recall': recall_score(y_val, y_pred, zero_division=0),
                            'F1': f1_score(y_val, y_pred, zero_division=0)
                        }
                        
                        fold_results.append(metrics)
                        
                    except Exception as e:
                        print(f"Erro no fold {fold}: {str(e)}")
                        continue
                
                if fold_results:
                    results[name] = self._calculate_mean_metrics(fold_results)
                
            except Exception as e:
                print(f"Erro ao treinar {name}: {str(e)}")
                continue
        
        return results

    def _plot_confusion_matrix(self, y_true, y_pred, model_name, fold):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} - Matriz de Confusão (Fold {fold})')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.savefig(f'{self.results_path}/{model_name.replace(" ", "_")}_cm_fold_{fold}.png')
        plt.close()

    def _plot_roc_curve(self, y_true, y_prob, model_name, fold):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_prob):.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        plt.title(f'{model_name} - Curva ROC (Fold {fold})')
        plt.legend()
        plt.savefig(f'{self.results_path}/{model_name.replace(" ", "_")}_roc_fold_{fold}.png')
        plt.close()

    def _calculate_mean_metrics(self, fold_results):
        mean_metrics = {}
        for metric in fold_results[0].keys():
            values = [fold[metric] for fold in fold_results]
            mean_metrics[metric] = {
                'Média': np.mean(values),
                'Desvio Padrão': np.std(values)
            }
        return mean_metrics

    def print_results(self, results):
        """Imprime os resultados de forma organizada"""
        print("\n=== Resultados Finais ===")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, values in metrics.items():
                print(f"{metric_name}:")
                print(f"  Média: {values['Média']:.4f}")
                print(f"  Desvio Padrão: {values['Desvio Padrão']:.4f}")

    def evaluate_model(self, model, X, y):
        """
        Realiza validação cruzada estratificada para o Dataset B (MiniBooNE)
        """
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        metrics = {
            'balanced_accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        print(f"\nIniciando validação cruzada (5 folds):")
        
        # Executar validação cruzada
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            # Separar dados
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Treinar e avaliar
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calcular métricas
            metrics['balanced_accuracy'].append(balanced_accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred))
            metrics['recall'].append(recall_score(y_val, y_pred))
            metrics['f1'].append(f1_score(y_val, y_pred))
            
            # Imprimir resultados do fold
            print(f"\nFold {fold}:")
            print(f"Balanced Accuracy: {metrics['balanced_accuracy'][-1]:.4f}")
            print(f"Precisão: {metrics['precision'][-1]:.4f}")
            print(f"Recall: {metrics['recall'][-1]:.4f}")
            print(f"F1-Score: {metrics['f1'][-1]:.4f}")
        
        # Calcular estatísticas finais
        results = {}
        for metric in metrics:
            results[metric] = {
                'Média': np.mean(metrics[metric]),
                'Desvio Padrão': np.std(metrics[metric])
            }
        
        return results

if __name__ == "__main__":
    # Carregar dados pré-processados
    from preprocessamento_b import DataPreprocessor
    
    # Preparar dados
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_analyze('data/MiniBooNE_B.txt')
    df_cleaned = preprocessor.clean_data(df)
    scaled_data, y = preprocessor.scale_data(df_cleaned)
    
    # Converter target para binário (0 ou 1)
    y = (y > 0.5).astype(int)
    
    # Usar dados normalizados para treinamento
    X = scaled_data['normalizer']
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()  # Converter para numpy array
    
    print("\nForma dos dados de entrada:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\nDistribuição das classes:")
    print(f"Classe 0: {np.sum(y == 0)} amostras")
    print(f"Classe 1: {np.sum(y == 1)} amostras")
    
    # Garantir que X e y são arrays numpy
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Treinar e avaliar modelos
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X, y)
    trainer.print_results(results)