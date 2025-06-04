import numpy as np
import os
from modelagem_b import ModelTrainer

def main():
    # Definir caminho base
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data')
    
    # Verificar se diretório existe
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Diretório de dados não encontrado: {data_path}")
    
    # Carregar dados preprocessados com caminho completo
    X_path = os.path.join(data_path, 'X_processed.npy')
    y_path = os.path.join(data_path, 'y_processed.npy')
    
    # Verificar se arquivos existem
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Arquivos de dados processados não encontrados")
    
    # Carregar dados
    X = np.load(X_path)
    y = np.load(y_path);
    
    trainer = ModelTrainer()
    
    print("=== Iniciando Validação Cruzada - Dataset B ===\n")
    
    results = {}
    for name, model_info in trainer.models.items():
        print(f"\nAvaliando modelo: {name}")
        results[name] = trainer.evaluate_model(model_info['model'], X, y)
    
    print("\n=== Resultados Finais ===\n")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, values in metrics.items():
            print(f"{metric}:")
            print(f"  Média: {values['Média']:.4f}")
            print(f"  Desvio Padrão: {values['Desvio Padrão']:.4f}")

if __name__ == "__main__":
    main()