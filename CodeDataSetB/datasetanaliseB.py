import pandas as pd

# 1. Carregar o dataset, pulando a primeira linha (ajuste skiprows se necessário)
df = pd.read_csv(
    "data/MiniBooNE_B.txt",
    sep=r'\s+',
    header=None,
    engine='python',
    skiprows=1  # Aumente se ainda houver erro: ex. skiprows=2
)

# 2. Informações iniciais
print("Formato original:", df.shape)
print("Colunas:", df.columns)
print(df.info())
print(df.head())

# 3. Remover colunas desnecessárias (caso queira remover algumas, edite a lista abaixo)
colunas_para_remover = []  # Exemplo: [0, 1] ou ['col_0', 'col_1']
df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns])

# 4. Tratar valores ausentes
df = df.dropna(thresh=int(0.8 * len(df.columns)))  # Remove linhas com muitos NaNs

# Preencher valores ausentes com mediana ou valor padrão
for coluna in df.columns:
    if df[coluna].dtype in ['float64', 'int64']:
        df[coluna] = df[coluna].fillna(df[coluna].median())
    elif df[coluna].dtype == 'object':
        df[coluna] = df[coluna].fillna(df[coluna].mode()[0])

# 5. Corrigir tipos de dados (se necessário)
for coluna in df.columns:
    if 'data' in str(coluna).lower():
        df[coluna] = pd.to_datetime(df[coluna], errors='coerce')
    elif df[coluna].dtype == 'object':
        try:
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce')
        except:
            pass

# 6. Remover duplicatas
df = df.drop_duplicates()

# 7. (Opcional) Filtrar dados irrelevantes
# Exemplo: df = df[df['col_5'] > 0]  # ajuste para sua coluna

# 8. Renomear colunas para facilitar visualização
df.columns = [f"col_{i}" for i in range(df.shape[1])]
df = df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))

# 9. Resetar o índice
df = df.reset_index(drop=True)

# 10. Salvar CSV limpo
df.to_csv("dataset_limpo.csv", index=False)

print("✅ Limpeza finalizada. Novo formato:", df.shape)
