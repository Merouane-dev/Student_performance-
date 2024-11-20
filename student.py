# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
import matplotlib.pyplot as plt

# Chargement du dataset
data = pd.read_csv('StudentPerformanceFactors.csv')

# Étape 1 : Traitement des valeurs manquantes
# Imputation des valeurs manquantes (mode pour catégoriques, médiane pour numériques)
imputer_categorical = SimpleImputer(strategy='most_frequent')
imputer_numeric = SimpleImputer(strategy='median')

# Colonnes avec des valeurs manquantes
data['Teacher_Quality'] = imputer_categorical.fit_transform(data[['Teacher_Quality']]).ravel()
data['Parental_Education_Level'] = imputer_categorical.fit_transform(data[['Parental_Education_Level']]).ravel()
data['Distance_from_Home'] = imputer_categorical.fit_transform(data[['Distance_from_Home']]).ravel()

# Étape 2 : Encodage des variables catégoriques
categorical_features = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level',
    'Distance_from_Home', 'Gender'
]

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(data[categorical_features])
encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Combinaison des données catégoriques encodées avec les données numériques
data_cleaned = pd.concat([data.drop(columns=categorical_features), encoded_features_df], axis=1)

# Étape 3 : Préparation du dataset pour l'arbre de décision
X = data_cleaned.drop(columns=['Exam_Score'])
y = data_cleaned['Exam_Score']

# Division des données en ensembles d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 4 : Construction et entrainement de l'arbre de décision
decision_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
decision_tree.fit(X_train, y_train)

# Étape 5 : Visualisation de l'arbre de décision
plt.figure(figsize=(20, 10))
tree.plot_tree(decision_tree, filled=True, feature_names=X.columns, rounded=True, fontsize=10)
plt.show()
