#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[3]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


countries = pd.read_csv("countries.csv")


# In[5]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
#countries.columns


# In[7]:


#countries.info()


# In[8]:


#countries.shape


# In[9]:


countries['Region'] = countries['Region'] = countries['Region'].apply(lambda y: y.strip())


# In[10]:


info = pd.DataFrame({'colunas':countries.columns, 
                      'tipo':countries.dtypes,
                      'Qtde valores NaN':countries.isna().sum(),
                      '% valores NaN':countries.isna().sum()/countries.shape[0],
                      'valores únicos por feature':countries.nunique()})
info = info.reset_index()


# In[11]:


#countries.describe()


# A tabela geral nos mostra diversas variáveis foram cadastradas com o tipo errada (são variáveis numéricas cadastradas como variáveis categóricas), comprovado pelo método describe que retorna valores apenas para variáveis numéricas. Provavelmente esse erro ocorreu devido ao uso do separador ',' ao invés do ponto.
# 

# In[12]:


#countries.dtypes


# In[13]:


#countries.sort_values(by='Region',inplace=True)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[14]:


def q1():
    regions = np.sort(countries['Region'].unique()).tolist()
    return list(regions)
    # Retorne aqui o resultado da questão 1.
    


# In[15]:


type(q1())


# In[16]:


for col in countries.select_dtypes(include='object').columns:
    countries[col] = countries[col].str.replace(',','.')


# In[17]:


change_category = ['Pop_density', 'Coastline_ratio', 'Net_migration',
       'Infant_mortality', 'Literacy', 'Phones_per_1000', 'Arable', 'Crops',
       'Other', 'Climate', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry',
       'Service']


# In[18]:


countries[change_category] = countries[change_category].astype('float64')
# countries.dtypes


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[19]:


# discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
# score_bins = discretizer.fit_transform(countries[["Pop_density"]])
# len(score_bins[score_bins>= 9.])


# In[20]:


def q2():
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    score_bins = discretizer.fit_transform(countries[["Pop_density"]])
    return int(len(score_bins[score_bins>= 9.]))
    # Retorne aqui o resultado da questão 2.


# In[21]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[22]:


# encoder = OneHotEncoder(sparse=False, dtype=np.int)
# enc = encoder.fit_transform(countries[['Region','Climate']].fillna({'Climate': 0}))
# enc.shape


# In[23]:


def q3():
    encoder = OneHotEncoder(sparse=False, dtype=np.int)
    enc = encoder.fit_transform(countries[['Region','Climate']].fillna({'Climate': 0}))
    
    return int(enc.shape[1])

    # Retorne aqui o resultado da questão 3.


# In[24]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[25]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[26]:


#fazendo a cópia do dataframe para não interferir nos valores do dataframe original
data_missing = countries.copy()
data_missing.head()


# In[27]:


data_missing.dtypes
#as duas colunas que não são do tipo int ou float é Country e Region, portanto vamos excluir elas.


# In[28]:


#Dataframe de teste 
df_test_country = pd.DataFrame([test_country], columns=data_missing.columns)

df_test_country


# In[29]:


pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])


# In[30]:


pipeline.fit(data_missing.drop(columns=['Country','Region'], axis=1))
test_pipeline = pipeline.transform(df_test_country.drop(columns=['Country','Region'], axis=1))
test_pipeline


# In[31]:


df_test = pd.DataFrame(test_pipeline, columns=df_test_country.drop(columns=['Country','Region'], axis=1).columns)
df_test


# In[32]:


def q4():
    return float(round(df_test['Arable'][0],3))    # Retorne aqui o resultado da questão 4.


# In[33]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[34]:


sns.boxplot(countries['Net_migration'], orient='vertical')


# In[35]:


countries.info()


# In[36]:


q1_quantile = countries['Net_migration'].quantile(0.25)
q3_quantile = countries['Net_migration'].quantile(0.75)
iqr = q3_quantile - q1_quantile

non_outlier_interval_iqr = [q1_quantile - 1.5 * iqr, q3_quantile + 1.5 * iqr]

#print(f"Faixa considerada \"normal\": {non_outlier_interval_iqr}")


# In[37]:


#calculo para achar os que estão fora da faixa normal
outliers_iqr_lower = countries['Net_migration'][countries['Net_migration'] < non_outlier_interval_iqr[0]]  
outliers_iqr_higher = countries['Net_migration'][countries['Net_migration'] > non_outlier_interval_iqr[1]]
qtd_outliers = len(outliers_iqr_lower)+len(outliers_iqr_higher)
#qtd_outliers


# In[38]:


qtd_dados = countries.shape[0]
#qtd_dados


# In[39]:


#print('Porcentagem de outliers: {}'.format((qtd_outliers*100)/qtd_dados))


# In[ ]:





# In[40]:


def q5():
    #Como temos 22% de outliers, precisaria de melhor analise para decidir como tratar, pois pode ter perda de informação.
    return (outliers_iqr_lower.shape[0], outliers_iqr_higher.shape[0], False)


# In[41]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[42]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[43]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(newsgroup.data)


# In[44]:


df_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
df_words['phone'].sum()


# In[45]:


def q6():
    df_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    return int(df_words['phone'].sum())
    # Retorne aqui o resultado da questão 4.


# In[46]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[47]:


tfidf_transformer = TfidfTransformer()

tfidf_transformer.fit(X)

newsgroups_tfidf = tfidf_transformer.transform(X)


# In[48]:


tfidf_vectorizer = TfidfVectorizer(use_idf=True)

newsgroups_tfidf_vectorized=tfidf_vectorizer.fit_transform(newsgroup.data)
newsgroups_tfidf_vectorized


# In[49]:


words_idx = sorted([vectorizer.vocabulary_.get('phone')])
df_phone = pd.DataFrame(newsgroups_tfidf[:, words_idx].toarray(), columns=np.array(vectorizer.get_feature_names())[words_idx])
df_phone['phone'].sum()



# In[50]:


def q7():
    return float(round(df_phone.sum(),3))

    # Retorne aqui o resultado da questão 4.


# In[51]:


q7()


# In[ ]:




