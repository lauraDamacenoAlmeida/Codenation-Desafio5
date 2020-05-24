#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[64]:


black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[51]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[8]:


mulheres = black_friday[black_friday['Age']=='26-35'].groupby(by='Gender')['User_ID'].count()
mulheres


# In[52]:


def q2():
    # Retorne aqui o resultado da questão 2.
    mulheres = black_friday[black_friday['Age']=='26-35'].groupby(by='Gender')['User_ID'].count()
    return int(mulheres[0])


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[53]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday.User_ID.nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[54]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    # Retorne aqui o resultado da questão 5.
    count=0
    for i in black_friday.values:
        for x in i:
            if pd.isna(x):
                count+=1
                break
    total_null_porcent = count/black_friday.shape[0]
    return total_null_porcent


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[13]:


black_friday.info()


# In[11]:


black_friday.isna().sum()


# In[9]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return int(black_friday.isna().sum().max())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


int(black_friday[black_friday['Product_Category_3'].notnull()]['Product_Category_3'].mode())


# In[10]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return int(black_friday[black_friday['Product_Category_3'].notnull()]['Product_Category_3'].mode())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[9]:


def q8():
    # Retorne aqui o resultado da questão 8.
    purschase_max = black_friday['Purchase'].max()
    purschase_min = black_friday['Purchase'].min()
    purschase_norm = (black_friday.Purchase - purschase_min)/(purschase_max-purschase_min)
    black_friday['Purchase_norm'] = purschase_norm
    return float(black_friday.Purchase_norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[12]:


def q9():
    # Retorne aqui o resultado da questão 9.
    mean_purchase = black_friday.Purchase.mean()
    std_purschace = black_friday.Purchase.std()
    black_friday['Purchase_pad'] = (black_friday.Purchase-mean_purchase)/std_purschace
    ax = black_friday[(black_friday['Purchase_pad']>=-1)&(black_friday['Purchase_pad']<=1)].shape[0]
    return ax


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[1]:


def q10():
    # Retorne aqui o resultado da questão 10.
    boolean = black_friday['Product_Category_2'].isna().equals(black_friday['Product_Category_2'].isna())
    return boolean


# In[ ]:




