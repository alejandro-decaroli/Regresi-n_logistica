## Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

## Analisis del datasets

df = pd.read_csv("/home/alejandro/proyectos/platzi/ia/machine_learning/Logisticas/dataframe.csv")

print(df.head())
print()
print(df.info()) # Se observa que la columna de TotalCharges es del tipo objeto, esto es rato ya que solo debe contener valores numericos

# Preprocesamiento del dataset

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce") # Con esto se transforma la columna a tipo numerico ignorando los errores

print()
print(df.isna().sum()) # 11 valores detectados como "na" no-a-number

df.dropna(inplace=True) # Se eliminan los valores na
df.drop(["customerID"], axis=1, inplace=True) # Se elimina la columna customerID por ser innecesaria

df_prepro = df.copy()
df_prepro = pd.get_dummies(df, drop_first=True, dtype=int)
print(df_prepro.info())

# Calcular la correlación con respecto a la variable de interés
correlations = df_prepro.corr()[["Churn_Yes"]].sort_values(by="Churn_Yes", ascending=True)

# Configurar el estilo de Seaborn para mejorar la apariencia del gráfico
sns.set(style='whitegrid')

# Crear un gráfico de barras ascendente (histograma) de las correlaciones
plt.figure(figsize=(10, 6))
sns.barplot(x=correlations["Churn_Yes"], y=correlations.index, palette='viridis')

# Configurar las etiquetas y el título del gráfico
plt.xlabel(f'Correlación con {"Churn_Yes"}')
plt.title(f'Correlación de variables con respecto a {"Churn_Yes"}')

# Mostrar el gráfico
plt.show()

scaler = MinMaxScaler()

df_prepro_scale = scaler.fit_transform(df_prepro)
df_prepro_scale = pd.DataFrame(df_prepro_scale)
df_prepro_scale.columns = df_prepro.columns
print(df_prepro_scale.head())






