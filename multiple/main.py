## Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler

## Dataset Analysis

df = pd.read_csv("/home/alejandro/proyectos/platzi/ia/machine_learning/Logisticas/multiple/Dry_Bean.csv")
pd.set_option('display.max_columns', None)
print(df.info())
print()
print(df.describe()) 

# Class Distribution
sns.countplot(df.Class) # Few BOMBAY data, hence discarding data from other classes for better distribution
plt.show()

df.drop_duplicates(inplace=True)
print(df.isnull().sum())

## Preprocessing

under_sampler = RandomUnderSampler(random_state=42)

x = df.drop(["Class"], axis=1)
y = df.Class

x_over, y_over = under_sampler.fit_resample(x, y)

# Verification
sns.countplot(y_over)
plt.show()

# Converting classes to numeric data

print(y_over.unique())
y_over.replace(["BARBUNYA","BOMBAY","CALI","DERMASON","HOROZ","SEKER","SIRA"], [1,2,3,4,5,6,7], inplace=True)

df_over = x_over
df_over["Class"] = y_over

plt.figure(figsize=(15,9))
sns.heatmap(df_over.corr(), annot=True) # Discarding ConvexArea, EquivDiameter, and MajorAxisLength for being highly correlated with area
plt.show()

x_over.drop(["ConvexArea","EquivDiameter","MajorAxisLength"], axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x_over, y_over, random_state=True, shuffle=True, test_size=0.2)

std = StandardScaler()

x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)

## Training

# Logistic model function
def logistic_model(C_, solver_, multiclass_):
    logistic_regression_model = LogisticRegression(random_state=42, solver=solver_, multi_class=multiclass_, n_jobs=-1, C=C_)
    return logistic_regression_model

# Function call
model = logistic_model(1,'saga','multinomial')
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))

# Iterative function call to test all parameters
multiclass = ['ovr','multinomial']
solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
scores = []
params=[]
for i in multiclass:
    for j in solver_list:
        try:
            model = logistic_model(1, j, i)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            params.append(i + '-' + j)
            accuracy = accuracy_score(y_test, predictions)
            scores.append(accuracy)
        except:
            None    

# Performance plot
fig = plt.figure(figsize=(10,10))
sns.barplot(x=params, y=scores).set_title('Beans Accuracy')
plt.xticks(rotation=0)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, predictions, labels=model.classes_) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='gray')
plt.show()
