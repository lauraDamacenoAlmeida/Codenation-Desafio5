#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

from loguru import logger


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


fifa = pd.read_csv("data.csv")


# In[4]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
fifa.head()


# In[6]:


fifa.columns


# In[7]:


fifa.describe()


# In[8]:


fifa.info()


# Temos alguns valores faltantes nas colunas

# In[9]:



faltantes = pd.DataFrame({'colunas':fifa.columns, 
                      'tipo':fifa.dtypes,
                      'Qtde valores NaN':fifa.isna().sum(),
                      '% valores NaN':fifa.isna().sum()/fifa.shape[0],
                      'valores únicos por feature':fifa.nunique()})
faltantes = faltantes.reset_index()
faltantes


# Como a quantidade de valores faltantes é mínima podemos excluí-los
# 

# In[10]:


fifa_drop = fifa.dropna()


# In[11]:


fifa_drop.shape


# In[12]:


pca = PCA(n_components=2) #Instanciando o PCA
projected = pca.fit_transform(fifa_drop)

print(f"Original shape: {fifa.shape}, projected shape: {projected.shape}")


# In[13]:


sns.scatterplot(projected[:, 0], projected[:, 1]);


# In[14]:


pca = PCA().fit(fifa_drop)

evr = pca.explained_variance_ratio_

evr


# In[15]:


g = plt.bar(range(len(evr)),evr)
plt.xlabel('Numero de compon')
plt.ylabel('Variância explicada');


# O dataset possui 37 variáveis, pelo gráfico de barras percebemos que as componentes principais 1 e 2 são capazes de explicar cerca de 80% do dataset.

# In[16]:


g = sns.lineplot(np.arange(len(evr)), np.cumsum(evr))
g.axes.axhline(0.95, ls="--", color="red")
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');


# Para chegar em um valor exato:

# In[17]:


cumulative_variance_ratio = np.cumsum(evr)
component_number = np.argmax(cumulative_variance_ratio >= 0.95) + 1 # Contagem começa em zero.

component_number


# Portanto cerca de 15 features são responsáveis por explicar 95% do dataset
# 
# 

# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[18]:


def q1():
    pca = PCA()
    # Retorne aqui o resultado da questão 1.
    pca.fit_transform(fifa_drop) 
    evr = pca.explained_variance_ratio_
    return float(round(evr[0],3))


# In[19]:


q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[20]:


def q2():
    pca = PCA().fit(fifa_drop)
    # Retorne aqui o resultado da questão 1.
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    component_number = np.argmax(cumulative_variance_ratio >= 0.95) + 1 # Contagem começa em zero.
    # Retorne aqui o resultado da questão 2.
    return int(component_number)


# In[21]:


q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# Usar dados centralizados implica que $\mu x = 0$ e a matriz de de covariância é dada por $C = \sum_{i=0}^N x_ix_i^T$.
# 
# 

# In[22]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[23]:


pca_2componentes = PCA(n_components=2)
pca_2componentes.fit(fifa_drop) 
np.dot(pca_2componentes.components_,x)


# In[24]:


def q3():
    pca_2componentes = PCA(n_components=2)
    pca_2componentes.fit(fifa_drop) 
    pc = np.dot(pca_2componentes.components_,x)
    return (round(pc[0],3),round(pc[1],3))
    # Retorne aqui o resultado da questão 3.


# In[25]:


q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# Uma feature interessante nesse dataset para ser considerada como target é a Overall que incorpora todos os atributos com ênfase em cada posição. Como pode ser verificado no link abaixo:
# 
# http://comufifa.blogspot.com/2012/05/fifa-entendendo-os-atributos.html
# 
# Informação encontrada na comunidade codenation

# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


# RFE retorna uma lista de boolean

# In[27]:


X = fifa_drop.drop(columns='Overall')
y = fifa_drop['Overall']

lr = LinearRegression()
rfe = RFE(lr, 5)
    
var = rfe.fit(X,y)
X.columns[var.support_]


# In[30]:


def q4():
    X = fifa_drop.drop(columns='Overall')
    y = fifa_drop['Overall']

    lr = LinearRegression()
    rfe = RFE(lr, 5)
    
    var = rfe.fit(X,y)
    return list(X.columns[var.support_])
    # Retorne aqui o resultado da questão 4.


# In[31]:


q4()


# In[ ]:




