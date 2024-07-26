#Importar bibliotecas
import numpy as np 
import pandas as pd 
import os
import seaborn as sns
from sklearn.metrics import accuracy_score
import plotly.express as px
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#Trabajando con datos
df=pd.read_csv("celiac_disease_lab_data.csv")

print(3*"\n",40*"=","Primeras 15 instancias",40*"=")
print(df.head(15))
print(3*"\n",40*"=","Información sobre datos",40*"=")
print(df.info())
print(3*"\n",40*"=","Forma del conjunto de datos",40*"=")
print(df.shape)

#Manejo de datos duplicados
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)
print(df.shape)

#Manejo de valores nulos
print(df.isnull().sum())

#Si Diabetes la respuesta es no, entonces no hay ningún tipo de diabetes, por lo que se asignará el valor tipo 0.
df["Diabetes Type"].fillna("Type 0",inplace=True)
print(df.head(15))

#Codificación de datos
print(df.dtypes)

encode=LabelEncoder()
df["Gender"]=encode.fit_transform(df["Gender"])
df["Diabetes"]=encode.fit_transform(df["Diabetes"])
df["Diabetes Type"]=encode.fit_transform(df["Diabetes Type"])
df["Diarrhoea"]=encode.fit_transform(df["Diarrhoea"])
df["Abdominal"]=encode.fit_transform(df["Abdominal"])
df["Short_Stature"]=encode.fit_transform(df["Short_Stature"])
df["Sticky_Stool"]=encode.fit_transform(df["Sticky_Stool"])
df["Weight_loss"]=encode.fit_transform(df["Weight_loss"])
df["Marsh"]=encode.fit_transform(df["Marsh"])
df["cd_type"]=encode.fit_transform(df["cd_type"])
df["Disease_Diagnose"]=encode.fit_transform(df["Disease_Diagnose"])
print(df.dtypes)

#Visualización de datos
#Mostrar distribución de datos
sns.pairplot(df)
plt.show()

plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),annot=True, cmap = "cool")
plt.show()
    
#Working on Features
Features=df.iloc[:,:-1]
Target=df.iloc[:,-1]

"""Elimine la diabetes porque si la respuesta es sí, el tipo de diabetes será tipo 1 o tipo 2, si la respuesta fue no, el tipo de diabetes 
será tipo 0, por lo que Puede reemplazar la diabetes con el tipo Diabetes.

Suelte diarrea, IgA, IgG e IgM según el mapa de calor que muestra correlación"""

Features.drop(["Diabetes","Diarrhoea","IgA","IgG","IgM"],axis=1,inplace=True)

Features_train,Features_test,Target_train,Target_test=train_test_split(Features,Target,train_size=0.75,random_state=21)

#Modelo de entrenamiento
Model=RandomForestClassifier(random_state=5,n_estimators=11)
Model.fit(Features_train,Target_train)

print(f"Precisión del entrenamiento = {Model.score(Features_train,Target_train)*100:.2f}\nPrecisión de las pruebas = {Model.score(Features_test,Target_test)*100:.2f}")

Prediction=Model.predict(Features)
score=accuracy_score(Target,Prediction)
print(f"Precisión de nuestro modelo. = {score*100:.2f}")

"""Alcance y objetivo del proyecto
Nuestro objetivo es predecir la probabilidad de que un paciente tenga enfermedad celíaca en función de algunas características médicas (clasificación binaria).

El conjunto de datos utilizado contiene 14 características y 2206 registros.

Las variables médicas utilizadas:
Edad
Género
Diabetes: sí, no
Tipo de diabetes: tipo 1, tipo 2, ninguna
Diarrea: grasa, acuosa, inflamatoria.
Abdominales: si, no
Estatura baja: PSS, variante, DSS
Taburete pegajoso: sí, no
Pérdida de peso: sí, no
Marsh: indica el nivel de linfocitos entre las células de la superficie del revestimiento intestinal: IEL (linfocitos intraepiteliales): pantano tipo 0, pantano tipo 3a, pantano tipo 1, pantano tipo 2, pantano tipo 3b, ninguno, pantano tipo 3c
cd_type: potencial, atípico, latente, silencioso, típico, ninguno
Los anticuerpos:

IgA
IgG
IgM
Enfoque del proyecto
Un primer vistazo al conjunto de datos sobre riesgos para la salud celíaca
Comprensión de datos
Preprocesamiento de datos
Manejo de valores faltantes
Codificación en caliente
Equilibrio de conjuntos de datos
Implementación del modelo
Bosque aleatorio
GridSearchCV para ajuste de parámetros
Prueba de modelo


Un primer vistazo al conjunto de datos sobre riesgos para la salud celíaca
Importaciones de biblioteca
Esta celda importa varias bibliotecas necesarias para el manejo de datos, la construcción de modelos y la visualización de datos."""

#Comprensión de datos
df = pd.read_csv('celiac_disease_lab_data.csv', sep=',', header=0)

print(df.shape)

print(df.head(5))

print(df.dtypes)

# ¿Cuáles son los tipos de valores en el campo?
print('Los valores de género son: ', df['Gender'].unique())
print('Los valores de diabetes son', df['Diabetes'].unique())
print('Los valores del tipo de diabetes son', df['Diabetes Type'].unique())
print('Los valores de diarrea son: ', df['Diarrhoea'].unique())
print('Los valores abdominales son', df['Abdominal'].unique())
print('Los valores de estatura corta son: ', df['Short_Stature'].unique())
print('Los valores de Sticky_Stool son: ', df['Sticky_Stool'].unique())
print('Los valores de pérdida de peso son: ', df['Weight_loss'].unique())
print('Los valores de Marsh son: ', df['Marsh'].unique())
print('Los valores de cd_type son: ', df['cd_type'].unique())
print('Los valores de Disease_Diagnose son: ', df['Disease_Diagnose'].unique())

#cuantos valores nulos hay
df.isnull().sum()

# comprobar el equilibrio de datos
df['Disease_Diagnose'].value_counts()

"""Notamos que el conjunto de datos está desequilibrado, ¡hay que equilibrarlo!

Preprocesamiento de datos
Manejo del valor faltante"""

df = df.fillna('none')
df.isnull().sum()

#Codificación en caliente
# Codificación de etiquetas para variables categóricas
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
        
df.dtypes

for column_name, label_encoder in label_encoders.items():
    print(f"Columna: {column_name}")
    numerical_labels = label_encoder.transform(label_encoder.classes_)
    original_values = label_encoder.inverse_transform(numerical_labels)
    for label, value in zip(numerical_labels, original_values):
        print(f"Etiqueta numérica: {label}, Valor original: {value}")
    print()
    
# Calcular la correlación entre la columna de destino 'Disease_Diagnose' y todas las características
df.corr()['Disease_Diagnose'].sort_values()

"""Conjunto de datos de equilibrio
Los casos de celíacos positivos multiplican por cinco el número de casos de celíacos negativos. Construir un modelo sobre un conjunto 
de datos tan desequilibrado produce resultados sesgados, por lo tanto, es de suma importancia equilibrar el conjunto de datos para un buen rendimiento del modelo.

Equilibrando nuestro datast: muestreo ascendente"""

df_majority = df[(df['Disease_Diagnose']==1)]
df_minority = df[(df['Disease_Diagnose']==0)]

df_minority.shape

df_majority.shape

# muestra superior de clase minoritaria
df_minority_upsampled = resample(df_minority,
                                 replace=True,    # muestra con reemplazo
                                 n_samples= 1843, # para coincidir con la clase mayoritaria
                                 random_state=42)  # resultados reproducibles

df_minority_upsampled.shape

# Combinar clase mayoritaria con clase minoritaria muestreada
df_upsampled = pd.concat([df_minority_upsampled, df_majority])

df_upsampled['Disease_Diagnose'].value_counts()

"""Modelado utilizando bosque aleatorio
Metodología de búsqueda en cuadrícula
Exploración sistemática de múltiples combinaciones de hiperparámetros.
Identifica valores óptimos mediante búsqueda exhaustiva.
División de datos
Esta celda divide el conjunto de datos en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo.

Training_valdiation establece el 70% de los datos.
Conjunto de prueba del 30% de los datos."""

X = df_upsampled.drop(columns=['Disease_Diagnose'])
y = df_upsampled['Disease_Diagnose']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba con una proporción de 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# comprobar el equilibrio de datos (y_train)
y_train.value_counts()

print ("X_train :", X_train.shape )
print ("y_train :", y_train.shape )

param_grid = {
    'n_estimators': [5 ],  # la cantidad de árboles en el bosque
    'max_depth': [10],  # la profundidad máxima de cada árbol suele ser 0 para evitar el sobreajuste
    'max_features': ['sqrt'],  # el número de características a considerar en cada división
    'min_samples_split': [50],  # el número mínimo de muestras necesarias para dividir un nodo
    'min_samples_leaf': [50],  # el número mínimo de muestras necesarias para estar en un nodo hoja
    'bootstrap': [True, False],  # si se deben utilizar muestras de arranque al construir árboles
    'criterion': ['gini', 'entropy'],  # la función para medir la calidad de una división
}

"""Configurando la búsqueda de cuadrícula para el árbol de decisión¶
Inicializa un DecisionTreeClassifier y configura GridSearchCV con:

El clasificador de árbol de decisión (clf) como estimador.
Una cuadrícula de parámetros (param_grid) para explorar.
Validación cruzada cuádruple (cv=4).
Puntuación f1 como métrica de puntuación.
Intenta cambiar la métrica de puntuación."""

# Crea una instancia de la clase RandomForestClassifier
rf = RandomForestClassifier()

# Crear una instancia de la clase GridSearchCV
grid_search_rf = GridSearchCV(estimator= rf ,
                              param_grid= param_grid,
                              cv=4,
                              scoring='f1',
                              n_jobs=-1,
                              verbose=2)

"""Prueba de modelo
Ejecutando búsqueda de cuadrícula
Ejecuta grid_search.fit en los datos de entrenamiento (X_train, y_train) para encontrar la configuración óptima de hiperparámetros basada 
en la cuadrícula definida."""

# Ajustar el objeto de búsqueda de cuadrícula a los datos
grid_search_rf.fit(X_train, y_train)

"""Puede llevar algún tiempo ajustar el modelo, ya que Grid Search intenta todas las combinaciones posibles de los valores de los 
parámetros que se especificaron en la cuadrícula de parámetros.

Recuperar los mejores parámetros del modelo desde la búsqueda de cuadrícula
Extrae best_params de grid_search, revelando la configuración óptima de hiperparámetros.
Asigna best_model como el mejor estimador encontrado mediante la búsqueda en cuadrícula.
Imprime best_params para mostrar los hiperparámetros seleccionados para el mejor modelo."""

# Mejores parámetros de RF:
best_model_rf = grid_search_rf.best_estimator_

best_params_rf = grid_search_rf.best_params_
print("Mejores parámetros de RF:", best_params_rf)

# Predecir en el conjunto de prueba
y_pred = grid_search_rf.best_estimator_.predict(X_test)

# Concatenar X_train e y_train
train_data = pd.concat([X_train, y_train], axis=1)

# Concatenar X_test e y_test
test_data = pd.concat([X_test, y_test], axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("Exactitud:", accuracy)

# Generar una matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión: \n", cm)

# Generar un informe de clasificación
cr = classification_report(y_test, y_pred)
print("\nInforme de clasificación: \n", cr)

#cuantos valores nulos hay
print(df.isnull().sum())

#cuáles son los tipos de valores en el campo
print(df['Diabetes Type'].unique())

#asumiendo que los nan son personas sin diabetes reemplazando el nan con un NA por no corresponde
df = df.fillna('NA')

# Comprobación para asegurarse de que los nulos se hayan ido
print(df.isnull().sum())

#comprueba los tipos de datos
print(df.dtypes)

print(df['Diabetes'].unique())

print(df.head())

df['Gender'] = df['Gender'].astype('string')
df['Diabetes Type'] = df['Diabetes Type'].astype('string')
df['Diarrhoea'] = df['Diarrhoea'].astype('string')
df['Short_Stature'] = df['Short_Stature'].astype('string')
df['Marsh'] = df['Marsh'].astype('string')
df['cd_type'] = df['cd_type'].astype('string')
df['Diabetes'] = df['Diabetes'].map({'Yes':True ,'no':False})
df['Abdominal'] = df['Abdominal'].map({'yes':True ,'no':False}) 
df['Sticky_Stool'] = df['Sticky_Stool'].map({'yes':True ,'no':False}) 
df['Weight_loss'] = df['Weight_loss'].map({'yes':True ,'no':False})
df['Disease_Diagnose'] = df['Disease_Diagnose'].map({'yes':True ,'no':False})

print(df.dtypes)

#necesitamos reemplazar los sustantivos en algunas funciones para obtener nombres de columna adecuados más adelante
print(df['Marsh'].unique())

df['Marsh'] = df['Marsh'].map({'none': 'marsh_none'}) 

df['cd_type'].unique()

df['cd_type'] = df['cd_type'].map({'none': 'cd_type_none'}) 

#en este punto quiero hacer una verificación de coordinación pero necesito convertir los datos de la cadena a booleanos.
df['Gender'].unique()

dummies = pd.get_dummies(df.Gender)
print(dummies)

merged = pd.concat([df, dummies], axis = 'columns')
print(merged)

merged.drop(['Gender'], axis='columns', inplace = True)

merged.rename(columns={'Diabetes Type': 'Diabetes_Type'}, inplace = True)

dummies1 = pd.get_dummies(merged.Diabetes_Type)
print(dummies1)
merged1 = pd.concat([merged, dummies1], axis = 'columns')
merged1.drop(['Diabetes_Type'], axis='columns', inplace = True)
print(merged1)

#Esto es un error de más arriba al no mirar hacia adelante en el análisis.
merged1.rename(columns={'NA': 'No_Diabetes'}, inplace = True)

#Quiero crear una función para hacer todo esto en una celda, pero primero hacerlo todo uno por uno.
dummies2 = pd.get_dummies(merged1.Diarrhoea)
print(dummies2)
merged2 = pd.concat([merged1, dummies2], axis = 'columns')
merged2.drop(['Diarrhoea'], axis='columns', inplace = True)

merged2['Short_Stature'].unique()
print(dummies2)

dummies3 = pd.get_dummies(merged2.Short_Stature)
print(dummies3)
merged3 = pd.concat([merged2, dummies3], axis = 'columns')
merged3.drop(['Short_Stature'], axis='columns', inplace = True)
print(merged3)

dummies4 = pd.get_dummies(merged3.Marsh)
print(dummies4)
merged4 = pd.concat([merged3, dummies4], axis = 'columns')
merged4.drop(['Marsh'], axis='columns', inplace = True)
print(merged4)

dummies5 = pd.get_dummies(merged4.cd_type)
print(dummies5)
merged5 = pd.concat([merged4, dummies5], axis = 'columns')
merged5.drop(['cd_type'], axis='columns', inplace = True)

merged5.dtypes

merged5.corr()['Disease_Diagnose'].sort_values()
print(merged5)

plt.figure(figsize=(16,14))
sns.heatmap(merged5.corr(), annot=True, cmap = "cool")
plt.show()

#quiero trabajar en una sola función para convertir todas las picaduras a booleanos en lugar de hacerlo una por una.
#1 Necesito una lista de las columnas a las que quiero realizar una codificación activa
#2 Revise cada columna necesaria para crear un muñeco. Ejemplo para una sola columna: dummies1 = pd.get_dummies(merged.Diabetes_Type)
#3 concatenación. Ejemplo para una sola columna:
#4 columna de caída. Ejemplo para una sola columna:
#5 siguiente columna
dummies1 = pd.get_dummies(merged.Diabetes_Type)
merged1 = pd.concat([merged, dummies1], axis = 'columns')
merged1.drop(['Diabetes_Type'], axis='columns', inplace = True)

#¿Quizás trabajar en un predictor de ML rudimentario?
#función única para convertir todas las cadenas a booleanos
