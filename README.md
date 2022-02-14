# Data-Cleaning
   # Missing value imputation by Mean, Median  
    # Method - Delete Rows & Column or Missing value imputation using Scikit-Learn
    
    
#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning¶
# 

# # Categorical Missing value imputation

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#load dataset
df = pd.read_csv(r"C:\Users\LEGION\Desktop\prac\train.csv")


# In[3]:


#categorical variable
cat_vars = df.select_dtypes(include='object')
cat_vars.head()


# In[4]:


cat_vars.isnull().sum()


# In[5]:


miss_val_per = cat_vars.isnull().mean()*100
miss_val_per


# In[6]:


drop_vars=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
cat_vars.drop(columns=drop_vars, axis=1, inplace=True)
cat_vars.shape


# In[7]:


isnull_per=cat_vars.isnull().mean()*100
miss_vars = isnull_per[isnull_per >0].keys()
miss_vars


# In[8]:


cat_vars['MasVnrType'].fillna('Missing')


# In[9]:


cat_vars['MasVnrType'].mode()


# In[10]:


cat_vars['MasVnrType'].value_counts()


# In[11]:


cat_vars['MasVnrType'].fillna(cat_vars['MasVnrType'].mode()[0])


# In[12]:


cat_vars['MasVnrType'].fillna(cat_vars['MasVnrType'].mode()[0]).value_counts()


# In[13]:


cat_vars_copy= cat_vars.copy()

for var in miss_vars:
    cat_vars_copy[var].fillna(cat_vars[var].mode()[0],inplace=True)
    print(var,"=",cat_vars[var].mode()[0])


# In[14]:


cat_vars_copy.isnull().sum().sum()


# In[15]:


plt.figure(figsize=(16,9))
for i,var in enumerate(miss_vars):
    plt.subplot(4,3,i+1)
    plt.hist(cat_vars_copy[var],label="Impute")
    plt.hist(cat_vars[var].dropna(),label="Original")
    plt.legend()


# In[16]:


df.update(cat_vars_copy)
df.drop(columns=drop_vars,inplace=True)


# In[17]:


df.select_dtypes(include='object').isnull().sum()


# In[ ]:




------------------------------------------------------------------------------

#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning¶
# 

# # Numerical Missing Value Imputation By Class

# In[2]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Load dataset
datset_path = r"https://drive.google.com/uc?export=download&id=1BiGZSedP4BIIuTbVTBodOhVgFImaz08c"
df = pd.read_csv(datset_path)


# In[4]:


df.shape


# In[5]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[6]:


df.head()


# In[7]:


missing_value_clm_gre_20 = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
df2_drop_clm = df.drop(columns=missing_value_clm_gre_20)
df2_drop_clm.shape


# In[8]:


df3_num=df2_drop_clm.select_dtypes(include=['int64','float64'])
df3_num.shape


# In[9]:


df3_num.isnull().sum()


# In[10]:


num_var_miss = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
df3_num[num_var_miss][df3_num[num_var_miss].isnull().any(axis=1)]


# In[11]:


#unique value in dataset
df['LotConfig'].unique()


# In[12]:


df[df.loc[:,'LotConfig'] == "Inside"]["LotFrontage"].replace(np.nan,df[df.loc[:,'LotConfig'] == "Inside"]["LotFrontage"].mean())


# In[14]:


#dataset update with this fun  df_copy.update
df_copy = df.copy()
for var_class in df['LotConfig'].unique():
    df_copy.update(df[df.loc[:,'LotConfig'] == var_class]["LotFrontage"].replace(np.nan,df[df.loc[:,'LotConfig'] == var_class]["LotFrontage"].mean()))


# In[15]:


df_copy.isnull().sum()


# In[16]:


df_copy = df.copy()
num_vars_miss = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_vars = ['LotConfig','MasVnrType','GarageType']
for cat_var, num_var_miss in zip(cat_vars,num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy.update(df[df.loc[:,cat_var] == var_class][num_var_miss].replace(np.nan,df[df.loc[:,cat_var] == var_class][num_var_miss].mean()))


# In[17]:


df_copy[num_vars_miss].isnull().sum()


# In[18]:


df_copy[df_copy[['MasVnrType']].isnull().any(axis=1)]


# In[19]:


df_copy[df_copy[['GarageType']].isnull().any(axis=1)]


# In[20]:


df_copy = df.copy()
num_vars_miss = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_vars = ['LotConfig','Exterior2nd','KitchenQual'] 
for cat_var, num_var_miss in zip(cat_vars,num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy.update(df[df.loc[:,cat_var] == var_class][num_var_miss].replace(np.nan,df[df.loc[:,cat_var] == var_class][num_var_miss].mean()))


# In[21]:


df_copy[num_vars_miss].isnull().sum()


# In[22]:


#data distribution
plt.figure(figsize=(10,10))
sns.set()
for i, var in enumerate(num_vars_miss):
    plt.subplot(2,2,i+1)
    sns.distplot(df[var], bins=20, kde_kws={'linewidth':8, 'color':'red'}, label="Original",)
    sns.distplot(df_copy[var], bins=20, kde_kws={'linewidth':5, 'color':'green'},label="Mean",)
    plt.legend()


# In[25]:


#median
df_copy_median = df.copy()
num_vars_miss = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_vars = ['LotConfig','Exterior2nd','KitchenQual']
for cat_var, num_var_miss in zip(cat_vars,num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy_median.update(df[df.loc[:,cat_var] == var_class][num_var_miss].replace(np.nan,df[df.loc[:,cat_var] == var_class][num_var_miss].median()))


# In[29]:


df_copy_median[num_vars_miss].isnull().sum()


# In[27]:


plt.figure(figsize=(10,10))
sns.set()
for i, var in enumerate(num_vars_miss):
    plt.subplot(2,2,i+1)
    sns.distplot(df[var], bins=20, kde_kws={'linewidth':8, 'color':'red'}, label="Original")
    sns.distplot(df_copy[var], bins=20, kde_kws={'linewidth':5, 'color':'green'},label="Mean")
    sns.distplot(df_copy_median[var], bins=20, kde_kws={'linewidth':3, 'color':'k'},label="Median")
    plt.legend()


# In[30]:


#Boxplot
for i, var in enumerate(num_vars_miss):
    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    sns.boxplot(df[var])
    plt.subplot(3,1,2)
    sns.boxplot(df_copy[var])
    plt.subplot(3,1,3)
    sns.boxplot(df_copy_median[var])


# In[ ]:




-------------------------------------------------------------------

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Data Cleaning
#Missing value imputation by Mean, Median


# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#from google drive load dataset
dataset_path = r"https://drive.google.com/uc?export=download&id=1BiGZSedP4BIIuTbVTBodOhVgFImaz08c"
df = pd.read_csv(dataset_path)


# In[3]:


df.shape


# In[4]:


#if u want see all data and columns
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[5]:


df.head(6)


# In[6]:


df.tail(6)


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


#how many percent percent null value 
missing_value_per = df.isnull().sum()/df.shape[0] * 100
missing_value_per


# In[10]:


#In data set 20% null value
missing_value_clm_gre_20 = missing_value_per[missing_value_per > 20].keys()
missing_value_clm_gre_20


# In[11]:


#drop cloumns
df2_drop_clm = df.drop(columns=missing_value_clm_gre_20)
df2_drop_clm.shape


# In[12]:


#Numerica data no categorical data
df3_num=df2_drop_clm.select_dtypes(include=['int64','float64'])
df3_num.head()
    


# In[13]:


plt.figure(figsize=(16,9))
sns.heatmap(df3_num.isnull())


# In[14]:


#In row null value present 
df3_num[df3_num.isnull().any(axis=1)]


# In[15]:


df3_num.isnull().sum()


# In[16]:


#in dataset null value in variable sum return 
missing_num_var = [var for var in df3_num.columns if df3_num[var].isnull().sum()>0]
missing_num_var


# In[17]:


#what is apply mean and median firstly check distribution check

plt.figure(figsize=(10,10))
sns.set()
for i, var in enumerate(missing_num_var):
    plt.subplot(2,2,i+1)
    sns.distplot(df3_num[var], bins=20, kde_kws={'linewidth':5, 'color':'#DC143C'})


# In[18]:


#every value with mean 
#0 return mean not any missing value
df4_num_mean = df3_num.fillna(df3_num.mean())
df4_num_mean.isnull().sum().sum()


# In[23]:


#origian or mean clean dataset plot
plt.figure(figsize=(10,10))
sns.set()
for i, var in enumerate(missing_num_var):
    plt.subplot(2,2,i+1)
    sns.distplot(df3_num[var], bins=20, kde_kws={'linewidth':8, 'color':'red'}, label="Original",)
    sns.distplot(df4_num_mean[var], bins=20, kde_kws={'linewidth':5, 'color':'green'},label="Mean",)
    plt.legend()


# In[21]:


df5_num_median = df3_num.fillna(df3_num.median())
df5_num_median.isnull().sum().sum()


# In[22]:


#original in mean median with clean 
#when outlier than diff mean median 
plt.figure(figsize=(10,10))
sns.set()
for i, var in enumerate(missing_num_var):
    plt.subplot(2,2,i+1)
    sns.distplot(df3_num[var], bins=20,hist=False, kde_kws={'linewidth':8, 'color':'red'}, label="Original",)
    sns.distplot(df4_num_mean[var], bins=20,hist=False, kde_kws={'linewidth':5, 'color':'green'},label="Mean",)
    sns.distplot(df5_num_median[var], bins=20,hist=False, kde_kws={'linewidth':3, 'color':'k'},label="Median",)
    plt.legend()


# In[24]:


#through fun box plot
for i, var in enumerate(missing_num_var):
    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    sns.boxplot(df[var])
    plt.subplot(3,1,2)
    sns.boxplot(df4_num_mean[var])
    plt.subplot(3,1,3)
    sns.boxplot(df5_num_median[var])


# In[25]:


df_concat = pd.concat([df3_num[missing_num_var],df4_num_mean[missing_num_var],df5_num_median[missing_num_var]], axis=1)


# In[26]:


#axis =1 mean return all nullvalue
df_concat[df_concat.isnull().any(axis=1)]


# In[ ]:




--------------------------------------------------------

#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# # Missing value imputation using Scikit-Learn

# # Different strategy for different variables(Numerical & Categorical) with Scikit-Learn

# In[2]:


import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[3]:


train = pd.read_csv(r"C:\Users\LEGION\Desktop\prac\train.csv")
test = pd.read_csv(r"C:\Users\LEGION\Desktop\prac\test.csv")
print("shape of train df = ",train.shape)
print("shape of test df = ",test.shape)


# In[4]:


X_train = train.drop(columns="SalePrice", axis=1)
y_train = train["SalePrice"]
X_test = test.copy()
print("Shape of X_train = ", X_train.shape)
print("Shape of y_train = ", y_train.shape)
print("Shape of X_test =", X_test.shape)


# # Missing value imputation

# In[5]:


isnull_sum = X_train.isnull().sum()
isnull_sum


# In[6]:


# finding the numerical variable which have mising value
num_vars = X_train.select_dtypes(include=["int64", "float64"]).columns
num_vars_miss = [var for var in num_vars if isnull_sum[var]>0]


# In[7]:


num_vars_miss


# In[8]:


# finding the categorical variable which have mising value
cat_vars = X_train.select_dtypes(include=["O"]).columns
cat_vars_miss = [var for var in cat_vars if isnull_sum[var]>0]
cat_vars_miss


# In[9]:


num_var_mean = ["LotFrontage"]
num_var_median = ['MasVnrArea', 'GarageYrBlt']
cat_vars_mode = ['Alley',
 'MasVnrType',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Electrical',
 'FireplaceQu',]
cat_vars_missing = ['GarageType',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PoolQC',
 'Fence',
 'MiscFeature']


# In[10]:


num_var_mean_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
num_var_median_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
cat_vars_mode_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
cat_vars_missing_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing"))])


# In[11]:


preprocessor = ColumnTransformer(transformers=[("mean_imputer", num_var_mean_imputer, num_var_mean),
                                ("median_imputer", num_var_median_imputer, num_var_median),
                               ("mode_imputer", cat_vars_mode_imputer, cat_vars_mode),
                               ("missing_imputer", cat_vars_missing_imputer,cat_vars_missing)])


# In[12]:


preprocessor.fit(X_train)


# In[13]:


preprocessor.transform


# In[14]:


preprocessor.named_transformers_["mean_imputer"].named_steps["imputer"].statistics_


# In[15]:


train["LotFrontage"].mean()


# In[16]:


preprocessor.named_transformers_["mode_imputer"].named_steps["imputer"].statistics_


# In[17]:


X_train_clean = preprocessor.transform(X_train)
X_test_clean = preprocessor.transform(X_test)


# In[18]:


X_train_clean


# In[19]:


preprocessor.transformers_


# In[20]:


X_train_clean_miss_var = pd.DataFrame(X_train_clean, columns=num_var_mean+num_var_median+cat_vars_mode+cat_vars_missing)


# In[21]:


X_train_clean_miss_var.head()


# In[22]:


X_train_clean_miss_var.isnull().sum().sum()


# In[23]:


train["Alley"].value_counts()


# In[24]:


X_train_clean_miss_var["Alley"].value_counts()


# In[25]:


X_train_clean_miss_var["MiscFeature"].value_counts()


# # Create Clean X_train DataFrame with call variables

# In[26]:


# no missing values variables index
remainder_vars_index = [0,
   1,
   2,
   4,
   5,
   7,
   8,
   9,
   10,
   11,
   12,
   13,
   14,
   15,
   16,
   17,
   18,
   19,
   20,
   21,
   22,
   23,
   24,
   27,
   28,
   29,
   34,
   36,
   37,
   38,
   39,
   40,
   41,
   43,
   44,
   45,
   46,
   47,
   48,
   49,
   50,
   51,
   52,
   53,
   54,
   55,
   56,
   61,
   62,
   65,
   66,
   67,
   68,
   69,
   70,
   71,
   75,
   76,
   77,
   78,
   79]


# In[27]:


# get no missing values variables name using there index
remainder_vars = [isnull_sum.keys()[var_index] for var_index in remainder_vars_index]
remainder_vars


# In[28]:


len(remainder_vars)


# In[29]:


# concatinate X_train_clean_miss_var df and remainder_vars
X_train =  pd.concat([X_train_clean_miss_var,train[remainder_vars]], axis=1)


# In[30]:


X_train.shape


# In[31]:


X_train.isnull().sum().sum()


# In[32]:


# Create test DataFrame with missing value imputed variables
X_test_clean_miss_var = pd.DataFrame(X_test_clean, columns=num_var_mean+num_var_median+cat_vars_mode+cat_vars_missing)
X_test_clean_miss_var.shape


# In[33]:


X_test_clean_miss_var.isnull().sum().sum()


# In[34]:


# concatinate X_test_clean_miss_var df and remainder_vars
X_test =  pd.concat([X_test_clean_miss_var,test[remainder_vars]], axis=1)
X_test.shape


# In[35]:


X_test.isnull().sum().sum()


# In[36]:


# 22 <= What is this, in X_test df still missing values as available but why 
#because we fill missing values in those columns which have missing value present in only X_train df
# Basicaly we get df then  find missing values variables then split df into X_train, X_test, y_train, y_test
# after that we fill missing value

# so if you have train and test df seperatly then first thing you should concatinate then find the missing 
# values variables it's is great strategy and carry on
# so you can try yourself


# In[37]:


isnull_sum_test = X_test.isnull().sum()
isnull_sum_test


# In[38]:


# finding the numerical variable which have mising value
num_vars_test = X_test.select_dtypes(include=["int64", "float64"]).columns
num_vars_miss_test = [var for var in num_vars_test if isnull_sum_test[var]>0]
num_vars_miss_test


# In[39]:


# finding the categorical variable which have mising value
cat_vars_test = X_test.select_dtypes(include=["O"]).columns
cat_vars_miss_test = [var for var in cat_vars_test if isnull_sum_test[var]>0]
cat_vars_miss_test


# In[40]:


# Hey it's time to do yourself

# If you are following this Jupyter NootBook file then please let me know in Comment box


# In[ ]:








