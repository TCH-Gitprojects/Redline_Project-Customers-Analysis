<div align="center">
  <center><h1>Redline Project : Customers Analysis</h1></center>
</div>

Projet de groupe réalisé dans le cadre de la formation en alternance Data Science de la Wild Code School. A partir d'une base de donnée client, faire un clustering et une étude des habitudes de la clientèle.

# Plusieurs objectifs pour ce projet : 

- Présentation de l'Analyse Exploratoire des Données (EDA) du Dataset
- Clustering pour Résumer les Segments de Clientèle
- Explication de la Pertinence des Clusters Créés
- Présentation du Travail dans une Application Django
- Dépôt GitHub

# EDA du dataset :

L'analyse a été réalisée avec Google Colab en utilisant du Python

## Aperçu des premières lignes de notre dataframe :

```python
main_df.head(10).style.background_gradient(cmap='Reds')
```
<img width="1336" alt="20240221100152" src="https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/a7af2014-89d5-40f3-b383-0fe29f88e9f3">

## Données statistiques du dataframe :
```python
df.describe().style.background_gradient(cmap='Reds')
```
<img width="1296" alt="20240221102619" src="https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/b4249dfd-ee28-4496-b72b-4d262e165fba">

## Nettoyage des données : Valeurs manquantes

### Visuel des NaN présents dans le dataframe
```python
def missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    Percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, Percentage], axis=1, keys=['Total', 'Percentage'])

missing_data(df).style.background_gradient(cmap='Reds')
```
<img width="399" alt="20240221101252" src="https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/f1f7efb7-1ce2-4593-9d56-2f29202bc725">

- On constate que la colonne Income possède 24 lignes sans données

### Suppression des NaN
```python
df.loc[(df['Income'].isnull() == True), 'Income'] = df['Income'].mean()

missing_data(df).style.background_gradient(cmap='Reds')
```
![20240221101733](https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/733aec4a-7578-4ef5-9a96-5a6f21a122c9)

- Les valeurs en NaN sont bien supprimées du dataframe
  
## Nettoyage des données : Doublons

### Recherche des doublons
```python
# Sélection de toutes les colonnes sauf 'ID'
subset = df.columns.drop('ID')
# Compte du nombre de doublons
df.duplicated(subset=subset).value_counts()

# Affichage de certaines lignes en doublon
duplicate_rows = df.duplicated(subset=subset,keep='first')
df[duplicate_rows].head()
```
![20240221102115](https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/867ccdb6-227f-457f-b7fb-0b66d44eef58)
- True = Doublons
### Suppressions des doublons
```python
# Suppression des doublons
df_2 = df.copy()
df_2.drop_duplicates(subset=subset, keep='first', inplace=True)

# Vérification
df_2.duplicated(subset=subset).value_counts()
```
![20240221102138](https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/40fdce84-cc7c-469e-888f-9d2e33e1b24d)
- Plus de doublons suite au travail de nettoyage

## Modifications/Suppressions des valeurs absurdes
### Remplacement des valeurs absurdes de la colonne Marital Status : 3 valeurs 'Alone' (seul) par 'Single' (célibataire), et suppression des 2 valeurs 'Absurd' et 'YOLO' qui sont négligeables.
```python
# Remplacement des valeurs 'Alone' par 'Single'
df_3 = df_2.copy()
mask_3 = (df_3.Marital_Status == 'Alone')
df_3.loc[mask_3, 'Marital_Status'] = 'Single'

# Suppression des lignes avec valeurs de Marial_Status 'Absurd' et 'YOLO'
mask_3 = (df_3.Marital_Status == 'Absurd') | (df_3.Marital_Status == 'YOLO')
# df_3[mask_3].index
df_3.drop(index=df_3[mask_3].index, inplace = True)
```
### Suppressions des colonnes inutiles (une seule valeur)
```python
df_3=df_3.drop(columns=["Z_CostContact", "Z_Revenue"],axis=1)
```

## Création de variables groupées et analyses :

### Kids :

- Nous avons choisi de regrouper les colonnes Kidhome et Teenhome en une seule variable nommée Kids

```python
df_4 = df_3.copy()
df_4['Kids'] = df_4['Kidhome'] + df_4['Teenhome']

# Camembert répartition du nb d'enfants dans le foyer
plt.figure(figsize=(4, 4))
df_4['Kids'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("Répartition du nombre de clients en fonction du nombre d'enfants du foyer")
plt.ylabel('')
plt.axis('equal')
plt.show()
```
<img width="659" alt="20240221103747" src="https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/07a52417-5733-4890-ad50-59cc77805e31">

### Expenses :

- Même logique utilisée pour regrouper les colonnes Mnt... en une seule et unique variable Expenses

```python
df_4['Expenses'] = df_4['MntWines'] + df_4['MntFruits'] + df_4['MntMeatProducts'] + df_4['MntFishProducts'] + df_4['MntSweetProducts'] + df_4['MntGoldProds']

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.distplot(df_4["Expenses"], color='red', ax=axes[0])
axes[0].set_title('Distribution de Expenses')

df_4["Expenses"].plot.box(color='red', ax=axes[1])
axes[1].set_title('Boîte à moustaches de Expenses')

plt.tight_layout()
plt.show()
```
<img width="1317" alt="20240221104320" src="https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/2532c315-5012-488b-b6c1-c24e7110d1cf">

### Nbr de visites par mois sur le site VS achats par mois sur le site

- Ici, pas de regroupement de colonnes mais une comparaison intéressante afin de vérifier si il existe une corrélation entre ces deux variables

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.distplot(df_4["NumWebVisitsMonth"], color='red', ax=axes[0])
axes[0].set_title('Distribution du nbr de visites sur le site par mois')

sns.distplot(df_4["NumWebPurchases"], color='red', ax=axes[1])
axes[1].set_title('Distribution du nbr dachats sur le site par mois')
```
<img width="1320" alt="20240221104721" src="https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/4781d5d9-b71d-4668-ab6a-cb2992fd9764">

### Situations familiales

- Par soucis de logique, nous créeons ici uniquement deux catégories : Single ou Relationship afin d'apprécier plus facilement la répartition de la clientèle

```python
df_5 = df_4.copy()
df_5['Relation_Status'] = df_5['Marital_Status']
df_5['Relation_Status'] = df_5['Relation_Status'].replace(['Married', 'Together'],'Relationship')
df_5['Relation_Status'] = df_5['Relation_Status'].replace(['Divorced', 'Widow'],'Single')

df_5['Relation_Status'].value_counts().plot(kind='bar',color = 'red',edgecolor = "black",linewidth = 1)
plt.title("Distribution des customers par situation relationnelle\n",fontsize=14)
plt.figure(figsize=(8,8))
```
<img width="573" alt="20240221105200" src="https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/bd05d363-24bd-4414-8edd-e35211aaa496">

### Income

- Ici, pas de regroupement nécessaire pour analyser la donnée

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.distplot(df_5["Income"], color='red', ax=axes[0])
axes[0].set_title('Distribution de Income')

df_5["Income"].plot.box(color='red', ax=axes[1])
axes[1].set_title('Boîte à moustaches de Income')

plt.tight_layout()
plt.show()
```
<img width="1318" alt="20240221105447" src="https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/c9fe4f9d-bc66-406d-9bc7-928aa5b671b2">

- On constate la présence d'un outlier dans la colonne Income que nous allons supprimer directement
```python
# Suppression des outliers de Income
df_6 = df_5.copy()
df_6.loc[df_6["Income"].idxmax()]
df_6.drop(df_6["Income"].idxmax(), inplace=True)
```

### Age

- Ici de la même façon que pour pour l'income, nous allons vérifier la présence d'outliers avant de produire le visuel d'analyse de la variable

```python
# Calcul de l'âge des clients
df_7 = df_6.copy()
df_7['Age']= 2024 - df_7['Year_Birth']

#Distribution de l'age
plt.figure(figsize=(4,3))
plt.boxplot(df_7["Age"],vert=False)
plt.title("Distribution de la colonne Age")
plt.xlabel("Age")
plt.show()
```
<img width="348" alt="20240221110026" src="https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/7956cc7a-9a04-4dac-a41d-02f13f047472">

- On constate également la présence d'outliers dans la colonne âge et nous allons donc les traiter afin d'assurer une cohérence dans nos données puis faire un regroupement par tranche d'âge afin d'obtenir un visuel plus facile à analyser

```python
# Suppression des clients de plus de 90ans (ceux de 120 ans doivent être décédés)
df_7 = df_7[(df_7['Age'] >= 20) & (df_7['Age'] <= 90)]

bins = range(20, df_7['Age'].max() + 10, 10)

# Création libellé perso
labels = [f"{i}-{i+10} ans" for i in range(20, df_7['Age'].max(), 10)]

# Utiliser pd.cut() avec les bins et les libellés personnalisés
df_7['Age_group'] = pd.cut(df_7['Age'], bins, right=False, labels=labels)

# Répartition des clients selon leur age
plt.figure(figsize=(8, 6))
df_7["Age_group"].value_counts().plot(kind='bar')
plt.title("Distribution des Ages des clients")
plt.xlabel("Age")
plt.ylabel("Nombre de clients")
plt.xticks(rotation=45)  # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
plt.show()
```
<img width="694" alt="20240221110246" src="https://github.com/TCH-Gitprojects/Redline_Project-Customers-Analysis/assets/127731574/634475a5-f9ad-4780-8ffe-d99c65e500be">
