from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class CrossValidation:
    def __init__(self, n_splits=5, random_state=42):
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
    def evaluate_model(self, model, X, y):
        """
        Realiza validação cruzada estratificada
        """
        # Converter para numpy array se necessário
        X = np.array(X)
        y = np.array(y)
        
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        print(f"\nIniciando validação cruzada ({self.skf.n_splits} folds):")
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y), 1):
            # Separar dados de treino e validação
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Treinar modelo
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calcular métricas
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred))
            metrics['recall'].append(recall_score(y_val, y_pred))
            metrics['f1'].append(f1_score(y_val, y_pred))
            
            print(f"\nFold {fold}:")
            print(f"Acurácia: {metrics['accuracy'][-1]:.4f}")
            print(f"Precisão: {metrics['precision'][-1]:.4f}")
            print(f"Recall: {metrics['recall'][-1]:.4f}")
            print(f"F1-Score: {metrics['f1'][-1]:.4f}")
        
        # Calcular médias e desvios
        results = {}
        for metric in metrics:
            results[metric] = {
                'Média': np.mean(metrics[metric]),
                'Desvio Padrão': np.std(metrics[metric])
            }
        
        return results