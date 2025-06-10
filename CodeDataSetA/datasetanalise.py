import pandas as pd

# 1. Carregar o dataset
df = pd.read_csv("data/taiwanese_bankruptcy_A.csv")

# 2. Informações iniciais
print("Formato original:", df.shape)
print("Colunas:", df.columns)
print(df.info())
print(df.head())

# 3. Remover colunas desnecessárias (modifique conforme seu caso)
colunas_para_remover = []  
df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns])

# 4. Tratar valores ausentes
df = df.dropna(thresh=int(0.8 * len(df.columns)))  # Remove linhas com muitos NaNs

# Preencher valores ausentes com mediana ou valor padrão
for coluna in df.columns:
    if df[coluna].dtype in ['float64', 'int64']:
        df[coluna] = df[coluna].fillna(df[coluna].median())
    elif df[coluna].dtype == 'object':
        df[coluna] = df[coluna].fillna(df[coluna].mode()[0])

# 5. Corrigir tipos de dados
for coluna in df.columns:
    if 'data' in coluna.lower():
        df[coluna] = pd.to_datetime(df[coluna], errors='coerce')
    elif df[coluna].dtype == 'object':
        try:
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
        except:
            pass

# 6. Remover duplicatas
df = df.drop_duplicates()

# 7. Filtrar dados irrelevantes (exemplos – ajuste conforme seu caso)
if 'valor_venda' in df.columns:
    df = df[df['valor_venda'] > 0]

# 8. Renomear colunas para facilitar visualização
df = df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))

# 9. Resetar o índice
df = df.reset_index(drop=True)

# 10. Salvar CSV limpo
df.to_csv("dataset_limpo.csv", index=False)

print("Limpeza finalizada. Novo formato:", df.shape)