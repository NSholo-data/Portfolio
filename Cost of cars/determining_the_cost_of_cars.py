#!/usr/bin/env python
# coding: utf-8

# # Определение стоимости автомобилей

# Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости. 
# 
# Заказчику важны:
# 
# - качество предсказания;
# - скорость предсказания;
# - время обучения.

# # План выполнения

# <br>1. Загрузка данных.<br>2. Изучение и предобработка данных. <br>
# -Заполнить пропущенные значения и обработать аномалии в столбцах. Если среди признаков имеются неинформативные, удалить их.<br>3. Подготовка выборок для обучения моделей.<br>4. Обучение разных моделей, одна из которых — LightGBM, как минимум одна — не бустинг. Для каждой модели пробуем разные гиперпараметры.<br>5. Анализ время обучения, время предсказания и качество моделей.<br>6. Опираясь на критерии заказчика, выбрать лучшую модель, проверить её качество на тестовой выборке.<br><br>
# Примечания:<br>- Для оценки качества моделей применяйте метрику RMSE.<br>- Значение метрики RMSE должно быть меньше 2500.<br>- Самостоятельно освоить библиотеку LightGBM и её средствами построить модели градиентного бустинга.<br>- Время выполнения ячейки кода Jupyter Notebook можно получить специальной командой. Найдите её.<br>- Модель градиентного бустинга может долго обучаться, поэтому измените у неё только два-три параметра.<br>- Если перестанет работать Jupyter Notebook, удалите лишние переменные оператором del.

# ## Подготовка данных

# In[1]:


import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import time
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor


# In[2]:


# открытие файла

try:
    df = pd.read_csv('/datasets/autos.csv')
except:
    df = pd.read_csv('autos.csv')


# In[3]:


# первые 5 строк

df.head()


# **Описание данных:**<br><br>
# **Признаки**<br><br>DateCrawled — дата скачивания анкеты из базы<br>VehicleType — тип автомобильного кузова<br>
# RegistrationYear — год регистрации автомобиля<br>Gearbox — тип коробки передач<br>Power — мощность (л. с.)<br>Model — модель автомобиля<br>Kilometer — пробег (км)<br>RegistrationMonth — месяц регистрации автомобиля<br>FuelType — тип топлива<br>Brand — марка автомобиля<br>Repaired — была машина в ремонте или нет<br>DateCreated — дата создания анкеты<br>NumberOfPictures — количество фотографий автомобиля<br>PostalCode — почтовый индекс владельца анкеты (пользователя)<br>LastSeen — дата последней активности пользователя<br><br>**Целевой признак**<br><br>Price — цена (евро)

# In[4]:


df.info()


# In[5]:


print('Размер исходных данных:', df.shape, 'Количество ячеек:', df.size)


# In[6]:


# проверка наличия явных дубликатов

print('Количество явных дубликатов', df.duplicated().sum())


# In[7]:


# удаление явных дубликатов

df = df.drop_duplicates()
print('Количество явных дубликатов', df.duplicated().sum())


# In[8]:


df.columns


# In[9]:


# приведем названия столбцов к корректному виду:
# добавим разделитель _
# приведем к нижнему регистру, удалим пробелы в начале и в конце

df = df.rename(columns = {'DateCrawled': 'Date_Crawled', 'VehicleType': 'Vehicle_Type', 
                          'RegistrationYear': 'Registration_Year', 
                          'RegistrationMonth': 'Registration_Month', 'FuelType': 'Fuel_Type',
                          'DateCreated': 'Date_Created', 'NumberOfPictures': 'Number_Of_Pictures',
                          'PostalCode': 'Postal_Code', 'LastSeen': 'Last_Seen'})
df.columns = df.columns.str.lower()
df.columns = df.columns.str.strip()


# In[10]:


df.columns


# In[11]:


# Количество пропусков в столбцах

df.isna().sum()


# In[12]:


df['number_of_pictures'].unique()


# In[13]:


# посмотрим подробнее столбцы с датами (date_crawled - дата скачивания анкеты из базы, date_created - дата создания анкеты, 
# и last_seen)

all_date = ['date_crawled', 'registration_year', 'date_created', 'last_seen']

for d in all_date:
    print(d)
    print(df[d].min())
    print(df[d].max())
    print()


# Данные в столбце registration_year' (год регистрации автомобиля) не совпадают с временным промежутком столбцов date_crawled - дата скачивания анкеты из базы, date_created - дата создания анкеты, и last_seen. Для registration_year' требуется обработка аномалий.

# Разберем столбцы:<br> - столбцы с датами: 'date_crawled' (дата скачивания анкеты из базы), 'registration_year' (год регистрации автомобиля), 'registration_month' (месяц регистрации автомобиля), 'date_created' (дата создания анкеты), 'last_seen' (дата последней активности пользователя) - из всех этих столбцов на цену влияет только **'registration_year' (год регистрации автомобиля)**, остальные можно удалить;<br> - столбец 'number_of_pictures' (количество фотографий автомобиля) имеет нулевые данные и, соответственно, не информативен; <br>'postal_code' (почтовый индекс владельца) не влияет на цену, удаляем

# In[14]:


# удаляем неинформативные столбцы
# 'number_of_pictures'
df = df.drop(['date_created', 'registration_month', 'date_crawled', 'number_of_pictures', 
           'postal_code', 'last_seen'], axis=1)


# In[15]:


df.columns


# In[16]:


df.info()


# In[17]:


df.head()


# In[18]:


print('Размер данных:', df.shape, 'Количество ячеек:', df.size)


# In[19]:


# уникальные значения столбца 'vehicle_type' (тип автомобильного кузова)

df['vehicle_type'].unique()


# In[20]:


# уникальные значения столбца 'gearbox' (тип коробки передач)

df['gearbox'].unique()


# In[21]:


# уникальные значения столбца 'fuel_type' (тип топлива)

df['fuel_type'].unique()


# In[22]:


# уникальные значения столбца 'model' (модель автомобиля), + удалим пробелы в названиях

df['model'] = df['model'].str.strip()
df['model'].sort_values().unique()


# In[23]:


# уникальные значения столбца 'brand' (марка автомобиля), + удалим пробелы в названиях

df['brand'] = df['brand'].str.strip()
df['brand'].sort_values().unique()


# In[24]:


df['repaired'].unique()


# In[25]:


df['repaired'] = df['repaired'].fillna('unknown')


# In[26]:


df['repaired'].unique()


# Исследуем числловые столбцы: price (цена), registration_year (год регистрации), power(мощность), kilometer(пробег, км).

# Исследуем целевой признак

# In[27]:


# целевой признак

df['price'].describe()


# In[28]:


# авто 500 и менее евро

df.loc[df['price'] <= 500]


# По архивным данным на 2016 год цены на данные авто превышали 500 евро, поэтому удаляем авто с нулевой ценой и менее 500 евро.

# In[29]:


# оставляем в данных только авто без нулевой цены и авто дороже 500 евро

df = df.loc[df['price'] > 500]
(df['price'].value_counts())


# In[30]:


# исследуем целевой признак

print(df['price'].value_counts())
print()
fig = go.Figure()
fig.add_trace(go.Box(y=df['price'], name='Цена (евро)'))
fig.update_layout(title='Диаграмма размаха столбца "price"')
fig.show()


# In[31]:


sns.displot(data=df['price'], kind='hist', rug=True, kde=True, height= 7, aspect= 1.5)
plt.title('График распределения "price" (цена)')
plt.show()


# In[32]:


df['price'].describe()


# In[33]:


print('Размер данных:', df.shape, 'Количество ячеек:', df.size)


# Удалили 11% строк. Распределение правосторонее унимодальное. Среднее значение отличается от медианы (4775 и 3300 соответственно). Данные выше max_quant_price не удаляем, так как там плавное равномерно убывают без разрывов. Строки ниже min_quant_price удалили, так как их маленький процент и в основном старые машины

# Исследуем год регистрации автомобиля

# In[34]:


df['registration_year'].sort_values().unique()


# In[35]:


print(df['registration_year'].sort_values(ascending=False).value_counts())
print()
fig = go.Figure()
fig.add_trace(go.Box(y=df['registration_year'], name='Год регистрации авто'))
fig.update_layout(title='Диаграмма размаха столбца "registration_year"')
fig.show()


# In[36]:


df['registration_year'].describe()


# Так как год скачивания анкеты 2016, и год последней активности пользователя также 2016, удаляем аномальные значения выше этой даты.
# Таже удалим авто старше 40 лет (до 1976 года).

# In[37]:


df = df.loc[((df['registration_year'] >= 1976) & (df['registration_year'] <= 2016))]


# In[38]:


sns.displot(data=df['registration_year'], kind='hist', rug=True, kde=True, height= 7, aspect= 1.5)
plt.title('График распределения "registration_year" (год регистрации авто)')
plt.show()


# In[39]:


fig = go.Figure()
fig.add_trace(go.Box(y=df['registration_year'], name='Год регистрации авто'))
fig.update_layout(title='Диаграмма размаха столбца "registration_year"')
fig.show()


# In[40]:


print('Размер данных:', df.shape, 'Количество ячеек:', df.size)


# Исследуем столбец "power" мощность (л.с.). Пропуски в нем отсутствуют

# In[41]:


df['power'].sort_values().unique()


# In[42]:


print(df['power'].sort_values(ascending=False).value_counts())
print()
fig = go.Figure()
fig.add_trace(go.Box(y=df['power'], name='Мощность, л.с.'))
fig.update_layout(title='Диаграмма размаха столбца "power"')
fig.show()


# In[43]:


df['power'].describe()


# In[44]:


sns.displot(data=df['power'], kind='hist', rug=True, kde=True, height= 5, aspect= 1.5)
plt.title('График распределения "power" (мощность)')
plt.show()


# В рейтинге легковых автомобилей по мощности в лошадиных силах есть авто с показателем выше 2000 л.с. Убираем выбросы с мощность более 2300 и авто с показателем 0.

# In[45]:


# удалим строки с нулевыми значениями и с мощностью более 220 л.с.

df = (df.loc[(df['power'] != 0) & (df['power'] < 2300)])


# In[46]:


fig = go.Figure()
fig.add_trace(go.Box(y=df['power'], name='Мощность, л.с.'))
fig.update_layout(title='Диаграмма размаха столбца "power"')
fig.show()


# In[47]:


sns.displot(data=df['power'], kind='hist', rug=True, kde=True, height= 7, aspect= 1.5)
plt.title('График распределения мощности "power"')
plt.show()


# In[48]:


print('Размер данных:', df.shape, 'Количество ячеек:', df.size)


# Удалено 22% строк от исходных данных

# Исследуем столбец "kilometer" пробег (км). Пропуски в нем отсутствуют

# In[49]:


# уникальные значения

df['kilometer'].sort_values().unique()


# In[50]:


# уникальные значения

df['kilometer'].describe()


# In[51]:


print(df['kilometer'].sort_values(ascending=False).value_counts())
print()
fig = go.Figure()
fig.add_trace(go.Box(y=df['kilometer'], name='Пробег (км)'))
fig.update_layout(title='Диаграмма размаха столбца "kilometer"')
fig.show()


# In[52]:


sns.displot(data=df['kilometer'], kind='hist', rug=True, kde=True, height= 7, aspect= 1.5)
plt.title('График распределения пробега "kilometer"')
plt.show()


# In[53]:


# Количество пропусков в столбцах

df.isna().sum()


# In[54]:


# Количество пропусков в процентах в столбцах

for i in df.columns:
    print('Процент пропусков в столбце:', i)
    print(f"{(df[i].isna().mean()):.2%}")


# Количество пропусков после удаления строк<br>
# **Итого пропусков в категориальных данных:** vehicle_type (тип автомобильного кузова) - 3372 (1,25%), gearbox (тип коробки передач) - 4019 (1,50%), model (модель автомобиля) - 8587 (3,20%), fuel_type (тип топлива) - 10305 (3,83%).

# Оставшиеся пропуски - это категориальные данные. У каждой модели может быть несколько типов кузова, тип топлива. Поэтому заполнить пропуски в типах кузова, типах топлива по бренду и модели не представляется возможным. Так как там не один вариант для замены, то заменим оставшиеся пропуски значением 'unknown'.

# In[55]:


df = df.fillna('unknown')


# In[56]:


# Количество пропусков в столбцах

df.isna().sum()


# In[57]:


df.duplicated().sum()


# In[58]:


df = df.drop_duplicates()


# In[59]:


print('Размер данных:', df.shape, 'Количество ячеек:', df.size)


# In[60]:


# количество и суммарные продажи по брендам
df_brand_sum = df.pivot_table(index='brand', values='price', aggfunc='sum').sort_values(by='price',
    ascending=False)
df_brand_count = df.pivot_table(index='brand', values='price', aggfunc='count').sort_values(by='price',
    ascending=False)
print('Сумма продаж авто по брендам', df_brand_sum)
print('Количество продаж авто по брендам', df_brand_count)


# In[61]:


df_brand_sum.plot(kind='bar', y='price', figsize=(15, 6))
plt.legend()
plt.title('Суммарные продажи авто по брендам')
plt.xlabel('Марка')
plt.ylabel('Суммарные продажи')
plt.show()


# In[62]:


df_brand_count.plot(kind='bar', y='price', figsize=(15, 6))
plt.legend()
plt.title('Количество продаж авто по брендам')
plt.xlabel('Марка')
plt.ylabel('Суммарные продажи')
plt.show()


# In[63]:


# количество и суммарные продажи по брендам
df_vehicle_type_sum = df.pivot_table(index='vehicle_type', values='price', 
                                     aggfunc='sum').sort_values(by='price',
    ascending=False)
df_vehicle_type_count = df.pivot_table(index='vehicle_type', values='price', 
                                       aggfunc='count').sort_values(by='price',
    ascending=False)
print('Сумма продаж авто по типу кузова', df_vehicle_type_sum)
print('Количество продаж авто по типу кузова', df_vehicle_type_count)


# In[64]:


df_vehicle_type_sum.plot(kind='bar', y='price', figsize=(15, 6))
plt.legend()
plt.title('Суммарные продажи авто по типу кузова')
plt.xlabel('Тип кузова')
plt.ylabel('Суммарные продажи')
plt.show()


# In[65]:


df_vehicle_type_count.plot(kind='bar', y='price', figsize=(15, 6))
plt.legend()
plt.title('Количество продаж авто по типу кузова')
plt.xlabel('Тип кузова')
plt.ylabel('Суммарные продажи')
plt.show()


# In[66]:


# количество и суммарные продажи по типу топлива
df_fuel_type_sum = df.pivot_table(index='fuel_type', values='price', 
                                     aggfunc='sum').sort_values(by='price',
    ascending=False)
df_fuel_type_count = df.pivot_table(index='fuel_type', values='price', 
                                       aggfunc='count').sort_values(by='price',
    ascending=False)
print('Сумма продаж авто по типу топлива', df_fuel_type_sum)
print('Количество продаж авто по типу топлива', df_fuel_type_count)


# In[67]:


df_fuel_type_sum.plot(kind='bar', y='price', figsize=(15, 6))
plt.legend()
plt.title('Суммарные продажи авто по типу топлива')
plt.xlabel('Тип топлива')
plt.ylabel('Суммарные продажи')
plt.show()


# In[68]:


df_fuel_type_count.plot(kind='bar', y='price', figsize=(15, 6))
plt.legend()
plt.title('Количество продаж авто по типу топлива')
plt.xlabel('Тип топлива')
plt.ylabel('Количество')
plt.show()


# In[69]:


# количество и суммарные продажи по типу коробки передач
df_gearbox_sum = df.pivot_table(index='gearbox', values='price', 
                                     aggfunc='sum').sort_values(by='price',
    ascending=False)
df_gearbox_count = df.pivot_table(index='gearbox', values='price', 
                                       aggfunc='count').sort_values(by='price',
    ascending=False)
print('Сумма продаж авто по типу коробки передач', df_gearbox_sum)
print('Количество продаж авто по типу коробки передач', df_gearbox_count)


# In[70]:


df_gearbox_sum.plot(kind='bar', y='price', figsize=(7, 5))
plt.legend()
plt.title('Суммарные продажи авто по типу коробки передач')
plt.xlabel('Тип коробки передач')
plt.ylabel('Суммарные продажи')
plt.show()


# In[71]:


df_gearbox_count.plot(kind='bar', y='price', figsize=(7, 5))
plt.legend()
plt.title('Количество продаж авто по типу коробки передач')
plt.xlabel('Тип коробки передач')
plt.ylabel('Количество')
plt.show()


# In[72]:


print('Размер полученных данных:', df.shape, 'Количество ячеек:', df.size)


# In[73]:


df.head()


# In[74]:


#df = df.copy().reset_index(drop=True)
print(df.info())
display(df.head())


# Алгоритмы машинного обучения принимают на вход числовые значения. Таким образом, мы должны манипулировать этими типами категорий, чтобы они были в правильном формате, приемлемом для этих алгоритмов обучения.

# Предоставлены исходные данные в размере 354 369 строк с индексацией 0-354368 и 16 столбцов. Типы столбцов корректные, названия столбцов и названия брендов автомобилей проиведены к корректному виду (приведены к нижнему регистру, удалены пробелы в начале и в конце наименования).
# В результате предобработки данных были удалены 4 явных дубликата и неинформатиные столбцы ( столбцы с датами: 'date_crawled' (дата скачивания анкеты из базы), 'registration_month' (месяц регистрации автомобиля), 'date_created' (дата создания анкеты), 'last_seen' (дата последней активности пользователя); столбец 'number_of_pictures' (количество фотографий автомобиля) имеет нулевые данные и, соответственно, не информативен; 'postal_code' (почтовый индекс владельца) не влияет на цену)<br> Удалены выбросы в столбцах 'price', 'registration_year', 'power', 'kilometr'. В категориальных переменнтых оставшиеся пропуски заменены на 'unknown'. В итоге количество строк сократилось на с 354369 до 232332, а количество столбцов на с 16 до 10.<br> В результате исследования выявлены топовые автомобили по брендам, по типам кузова, по типу топлива и по типу коробки передач.<br><br> Для дальнейшего обучения моделей постребуется кодировка категориальных признаков. Для преобразования данных выберем методику порядкового кодирования Original Encoding для работы с моделями, основанными на деревьях - дерево решений, случайный лес, градиентный бустинг. Для линейных моделей используем прямое кодирование One Hot Encoding. 

# ## Обучение моделей

# Кодирование данных:
# 1. Базовые - features
# 2. Кодировка Original Encoding - ord_features
# 3. Кодировка One Hot Encoding - ohe_features

# In[75]:


# признаки
features = df.drop(['price'], axis=1)
# целевой признак
target = df['price']
print(features.shape)
print(target.shape)


# In[76]:


features.head(3)


# In[77]:


features.isna().sum()


# In[78]:


target.isna().sum()


# In[79]:


# Определим random_state
RANDOM_STATE=12345


# In[80]:


# определяем категориальные признаки

cat_features = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'gearbox', 'brand']


# In[81]:


# функция для оценки важности признаков

def importances(model, features):
    features=features.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(11, 5))
    plt.title('Важность функции')
    plt.barh(range(len(indices)), importances[indices], color='#6495ED', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Относительная важность')
    plt.show()
        
    feat_importances = pd.Series(importances, index=features).sort_values(ascending=False)
    feat_importances = feat_importances.to_frame()
    feat_importances.columns=['Относительная важность']
    display(feat_importances)


# ### LinearRegression

# Для LinearRegression используем One-hot-encoding

# In[82]:


ohe_df = df.copy()


# In[83]:


#OHE кодирование: One-hot-encoding: получение дамми-признаков (для лин моделей)

ohe_df = pd.get_dummies(ohe_df, drop_first=True)
ohe_df.shape


# In[84]:


# признаки
ohe_features = ohe_df.drop(['price'], axis=1)

# целевой признак
ohe_target = ohe_df['price']
print(ohe_features.shape)
print(ohe_target.shape)


# In[85]:


ohe_features.head(3)


# In[86]:


# делим данные на обучающую train и валидационную + тестовую выборки (valid_test)
# в соотношении 60/40
ohe_features_train, ohe_features_valid_test, ohe_target_train, ohe_target_valid_test = train_test_split(
    ohe_features, ohe_target, test_size=0.4, random_state=RANDOM_STATE)


# In[87]:


# делим данные валидационную + тестовую выборки (valid_test) в соотношении 50/50
ohe_features_valid, ohe_features_test, ohe_target_valid, ohe_target_test = train_test_split(
    ohe_features_valid_test, ohe_target_valid_test, test_size=0.5, random_state=RANDOM_STATE)


# In[88]:


print('Размер обучающей выборки:')
print(ohe_features_train.shape)
print(ohe_target_train.shape)
print('Размер тестовой выборки:')
print(ohe_features_test.shape)
print(ohe_target_test.shape)
print('Размер валидационной выборки:')
print(ohe_features_valid.shape)
print(ohe_target_valid.shape)


# Масштабирование признаков:

# In[89]:


# список всех численных признаков 
list_ohe_fearures = list(ohe_features.columns)
print(list_ohe_fearures)
print(len(list_ohe_fearures))


# In[90]:


numeric = list_ohe_fearures
print(numeric)
print(len(numeric))


# In[91]:


scaler = StandardScaler()
scaler.fit(ohe_features_train[numeric])


# In[92]:


ohe_features_train[numeric] = scaler.transform(ohe_features_train[numeric])


# In[93]:


ohe_features_test[numeric] = scaler.transform(ohe_features_test[numeric])
ohe_features_valid[numeric] = scaler.transform(ohe_features_valid[numeric])


# In[94]:


get_ipython().run_cell_magic('time', '', "\nmodel_LR = LinearRegression()\nstart_model_LR = time.time() # время начала \n# обучим модель линейной регрессии\nmodel_LR.fit(ohe_features_train, ohe_target_train)\nend_model_LR = time.time() # время окончания\nfit_time_model_LR = end_model_LR - start_model_LR\nprint('Время обучения model_LR', fit_time_model_LR)\n")


# In[95]:


get_ipython().run_cell_magic('time', '', "\nstart_model_LR = time.time() # время начала \n# прогноз\npredictions_LR = model_LR.predict(ohe_features_valid)\nend_model_LR = time.time()\npredict_time_model_LR = end_model_LR - start_model_LR\nprint('Время предсказания model_LR', predict_time_model_LR)\n")


# In[96]:


get_ipython().run_cell_magic('time', '', "\n#найдем квадратичную среднюю ошибку (MSE)\n#найдем корень из квадратичной средней ошибки (RMSE)\nmse_LR = mean_squared_error(ohe_target_valid, predictions_LR)\nrmse_LR = mse_LR ** 0.5\nprint('mse_LR', mse_LR)\nprint('rmse_LR', rmse_LR)\n")


# ### DecisionTreeRegressor

# In[97]:


df = df.copy()


# In[98]:


# признаки
features = df.drop(['price'], axis=1)
# целевой признак
target = df['price']
print(features.shape)
print(target.shape)


# In[99]:


# делим данные на обучающую train и валидационную + тестовую выборки (valid_test)
# в соотношении 60/40
features_train, features_valid_test, target_train, target_valid_test = train_test_split(
    features, target, test_size=0.4, random_state=12345)


# In[100]:


# делим данные валидационную + тестовую выборки (valid_test) в соотношении 50/50
features_valid, features_test, target_valid, target_test = train_test_split(
    features_valid_test, target_valid_test, test_size=0.5, random_state=RANDOM_STATE)


# In[101]:


features_train.head(3)


# In[102]:


print('Размер обучающей выборки:')
print(features_train.shape)
print(target_train.shape)
print('Размер тестовой выборки:')
print(features_test.shape)
print(target_test.shape)
print('Размер валидационной выборки:')
print(features_valid.shape)
print(target_valid.shape)


# In[103]:


#кодирую признаки через OE:
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# обучение Encoder (на тренировочной выборке)
encoder.fit(features_train)


# In[104]:


# преобразуем данные функцией transform()
ord_features_train = encoder.transform(features_train)
ord_features_test = encoder.transform(features_test)
ord_features_valid = encoder.transform(features_valid)


# In[105]:


# реализуем обновленный датафрейм. Тип данных int
ord_features_train = pd.DataFrame(ord_features_train, columns = features_train.columns, dtype=int)
ord_features_valid = pd.DataFrame(ord_features_valid, columns = features_valid.columns, dtype=int)
ord_features_test = pd.DataFrame(ord_features_test, columns = features_test.columns, dtype=int)


# In[106]:


# результат
display(ord_features_train.head(3))


# In[107]:


# результат
display(ord_features_test.head(3))


# In[108]:


# результат
display(ord_features_valid.head(3))


# In[109]:


ord_features_train.isna().sum().sum()


# In[110]:


ord_features_test.isna().sum().sum()


# In[111]:


ord_features_valid.isna().sum().sum()


# In[112]:


ss = StandardScaler()
ss.fit(ord_features_train)
# стандартизируем данные
scale_features_train = ss.transform(ord_features_train)
# реализуем обновленный датафрейм. Тип данных int
scale_features_train = pd.DataFrame(scale_features_train, columns = ord_features_train.columns)
scale_features_train.head(3)

scale_features_valid = ss.transform(ord_features_valid)
scale_features_valid = pd.DataFrame(scale_features_valid, columns = ord_features_valid.columns)
scale_features_test = ss.transform(ord_features_test)
scale_features_test = pd.DataFrame(scale_features_test, columns = ord_features_test.columns)


# In[113]:


get_ipython().run_cell_magic('time', '', '\n# модель дерева решений на OE выборках\n\nbest_rmse_DTR = 5000\nbest_depth_DTR = 0\ndepths = []\nresults = []\n\nfor depth in range(1, 20):\n    model_DTR = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)\n    model_DTR.fit(scale_features_train, target_train)\n    predictions_DTR = model_DTR.predict(scale_features_valid)\n    mse_DTR = mean_squared_error(target_valid, predictions_DTR)\n    rmse_DTR = mse_DTR**0.5\n    print(\'Глубина:\', depth)\n    print(\'MSE для решающего дерева:\', mse_DTR)\n    print(\'RMSE для решающего дерева:\', rmse_DTR)\n    depths.append(depth)\n    results.append(rmse_DTR)\n    plt.plot(depths, results)\n    plt.title("Зависимость RMSE от глубины дерева")\n    plt.xlabel("Глубина дерева")\n    plt.ylabel("RMSE")\n    plt.grid()\n    plt.show\n    \n    if rmse_DTR < best_rmse_DTR:\n        best_rmse_DTR = rmse_DTR\n        best_depth_DTR = depth\nprint()\nprint(\'Итог DecisionTreeRegressor, model_DTR\')\nprint(\'RMSE для решающего дерева:\', best_rmse_DTR)\nprint(\'Глубина дерева:\', best_depth_DTR)\n')


# In[114]:


importances(model_DTR, scale_features_valid)


# In[115]:


get_ipython().run_cell_magic('time', '', "\nstart_model_DTR = time.time() # время начала \n# модель с лучшим параметром\nmodel_DTR = DecisionTreeRegressor(max_depth=best_depth_DTR, random_state=RANDOM_STATE)\n# обучение\nmodel_DTR.fit(scale_features_train, target_train)\n\nend_model_DTR = time.time() # время окончания\nfit_time_model_DTR = end_model_DTR - start_model_DTR\nprint('Время обучения model_DTR', fit_time_model_DTR)\n")


# In[116]:


get_ipython().run_cell_magic('time', '', "start_model_DTR = time.time()\n# прогноз\npredictions_DTR = model_DTR.predict(scale_features_valid)\nend_model_DTR = time.time() # время окончания\npredict_time_model_DTR = end_model_DTR - start_model_DTR\nprint('Время предсказания model_DTR', predict_time_model_DTR)\n")


# In[117]:


get_ipython().run_cell_magic('time', '', "\nmse_DTR = mean_squared_error(target_valid, predictions_DTR)\nrmse_DTR = mse_DTR ** 0.5\nprint('MSE для решающего дерева:', mse_DTR)\nprint('RMSE для решающего дерева:', rmse_DTR)\n")


# ### RandomForestRegressor

# In[118]:


get_ipython().run_cell_magic('time', '', "\nbest_model_RFR = 10\nbest_rmse_RFR = 5000\nbest_depth_RFR = 0\nbest_est_RFR = 0\nfor est in range(5, 50, 5):\n    for depth in range(2, 20, 2):\n        model_RFR = RandomForestRegressor(random_state=RANDOM_STATE,\n                                         n_estimators=est, max_depth=depth)\n        model_RFR.fit(scale_features_train, target_train)\n        prediction_RFR = model_RFR.predict(scale_features_valid)\n        mse_RFR = mean_squared_error(target_valid, prediction_RFR)\n        rmse_RFR = mean_squared_error(target_valid, prediction_RFR)**0.5\n        if rmse_RFR < best_rmse_RFR:\n            best_model_RFR = model_RFR\n            best_rmse_RFR = rmse_RFR\n            best_est_RFR = est\n            best_depth_RFR = depth\nprint()\nprint('Итог RandomForestRegressor')\nprint('RMSE для случайного леса:', best_rmse_RFR)\nprint('Количество деревьев', best_est_RFR)\nprint('Глубина дерева:', best_depth_RFR)\n")


# In[119]:


get_ipython().run_cell_magic('time', '', "\nstart_model_RFR = time.time() # время начала \n# обучаем модель с лучшими параметрами\nmodel_RFR = RandomForestRegressor(random_state=RANDOM_STATE,\n                                  n_estimators=best_est_RFR, max_depth=best_depth_RFR)\nmodel_RFR.fit(scale_features_train, target_train)\n\nend_model_RFR = time.time() # время окончания\nfit_time_model_RFR = end_model_RFR - start_model_RFR\nprint('Время обучения model_RFR', fit_time_model_RFR)\n")


# In[120]:


get_ipython().run_cell_magic('time', '', "\nstart_model_RFR = time.time() # время начала \n# прогноз\nprediction_RFR = model_RFR.predict(scale_features_valid)\nend_model_RFR = time.time() # время окончания\npredict_time_model_RFR = end_model_RFR - start_model_RFR\nprint('Время предсказания model_RFR', predict_time_model_RFR)\n")


# In[121]:


get_ipython().run_cell_magic('time', '', "mse_RFR = mean_squared_error(target_valid, prediction_RFR)\nrmse_RFR = mse_RFR ** 0.5\nprint('Итог RandomForestRegressor')\nprint('RMSE для случайного леса:', best_rmse_RFR)\nprint('Количество деревьев', best_est)\nprint('Глубина дерева:', best_depth)\n")


# In[122]:


importances(model_RFR, scale_features_valid)


# #### с RandomizedSearchCV

# In[123]:


get_ipython().run_cell_magic('time', '', "# подбираем лучшие гиперпараметры  n_estimators, max_depth, min_samples_leaf с RandomizedSearchCV,\n# выбираем метрику настройки RMSE\nn_estimators = range(1, 100)\nmax_depth = range(1, 50)\nmin_samples_leaf = range(1, 13)\n  \nhyperparameters = dict(\n    max_depth = max_depth, \n    min_samples_leaf = min_samples_leaf, \n    n_estimators = n_estimators\n)\nRandomForest = RandomForestRegressor(random_state = RANDOM_STATE)\n\nrandomizedsearch = RandomizedSearchCV(\n    RandomForest, hyperparameters, random_state = RANDOM_STATE, scoring='neg_root_mean_squared_error')\nRFR_best_model_random = randomizedsearch.fit(scale_features_train, target_train)\nprint(RFR_best_model_random.best_estimator_)\n")


# In[124]:


# выводим результаты кросс-валидации
results = pd.DataFrame(RFR_best_model_random.cv_results_)
print('RandomForestRegressor cv mean test score:', abs(results.mean_test_score.mean()).round(2))


# In[125]:


get_ipython().run_cell_magic('time', '', "\nstart_model_RFR_SearchCV = time.time() # время начала \n# обучаем лучшую модель\nmodel_RFR_SearchCV = RandomForestRegressor(\n    random_state=RANDOM_STATE, max_depth=35, n_estimators=86)\nmodel_RFR_SearchCV.fit(scale_features_train, target_train)\nend_model_RFR_SearchCV = time.time() # время окончания\nfit_time_model_RFR_SearchCV = end_model_RFR_SearchCV - start_model_RFR_SearchCV\nprint('Время обучения model_RFR_SearchCV', fit_time_model_RFR_SearchCV)\n")


# In[126]:


get_ipython().run_cell_magic('time', '', "\nstart_model_RFR_SearchCV = time.time() # время начала \n# делаем прогноз с лучшей моделью, считаем RMSE\npredictions_RFR_SearchCV = model_RFR_SearchCV.predict(scale_features_valid)\nend_model_RFR_SearchCV = time.time() # время окончания\npredict_time_model_RFR_SearchCV = end_model_RFR_SearchCV - start_model_RFR_SearchCV\nprint('Время предсказания model_RFR_SearchCV', predict_time_model_RFR_SearchCV)\n")


# In[127]:


get_ipython().run_cell_magic('time', '', "\nmse_RFR_SearchCV = mean_squared_error(target_valid, predictions_RFR_SearchCV)\nrmse_RFR_SearchCV = mse_RFR_SearchCV ** 0.5\nprint('RMSE_RFR_SearchCV при RandomizedSearchCV по модели RandomForestRegressor_SearchCV')\nprint(rmse_RFR_SearchCV)\n")


# In[128]:


importances(model_RFR_SearchCV, scale_features_train)


# ### CatBoostRegressor

# Это библиотека с открытым исходным кодом, основанная на деревьях решений повышения градиента

# In[129]:


# создаем список категориальных параметров
cat_features = features_train.select_dtypes(include=['object'])
print(list(cat_features))


# In[130]:


cat_features = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'brand', 'repaired']


# In[131]:


#%%time

start_model_CBR = time.time() # время начала 
# обучение
model_CBR = CatBoostRegressor(loss_function='RMSE', random_state=RANDOM_STATE)
model_CBR.fit(features_train, target_train, cat_features, verbose=10)
end_model_CBR = time.time() # время окончания

fit_time_model_CBR = end_model_CBR - start_model_CBR
print('Время обучения model_CBR', fit_time_model_CBR)


# In[132]:


get_ipython().run_cell_magic('time', '', "\nstart_model_CBR = time.time() # время начала \n# прогноз\nprobabilities_valid_CBR = model_CBR.predict(features_valid)\nend_model_CBR = time.time() # время окончания\npredict_time_model_CBR = end_model_CBR - start_model_CBR\nprint('Время предсказания model_CBR', predict_time_model_CBR)\n")


# In[133]:


get_ipython().run_cell_magic('time', '', '\nrmse_CBR = mean_squared_error(target_valid, probabilities_valid_CBR)**0.5\nprint()\nprint(rmse_CBR)\n')


# In[134]:


importances(model_CBR, features_train)


# ### LightGBMRegressor

# LightGBM — это фреймворк, который предоставляет реализацию деревьев принятия решений с градиентным бустингом

# In[135]:


# выделим признаки и целевой признак
target = df['price'].copy()
features = df.drop('price', axis=1).copy()

# переведем категориальные признаки в тип category, который требуется для LightGBM

for i in features.columns:
    col_type = features[i].dtype
    if col_type == 'object':
        features[i] = features[i].astype('category')

# делим данные на обучающую train и валидационную + тестовую выборки (valid_test)
# в соотношении 60/40
features_train, features_valid_test, target_train, target_valid_test = train_test_split(
    features, target, test_size=0.4, random_state=12345)    
# делим данные валидационную + тестовую выборки (valid_test) в соотношении 50/50
features_valid, features_test, target_valid, target_test = train_test_split(
    features_valid_test, target_valid_test, test_size=0.5, random_state=12345)

# проверим размер выборок
print(features_train.shape)
print(features_test.shape)


# In[136]:


print(features_train.dtypes)


# #### LightGBMRegressor без подбора гиперпараметров

# In[137]:


get_ipython().run_cell_magic('time', '', "\n# построим модель без подбора гиперпараметров\nstart_model_LGBMR = time.time() # время начала \nmodel_LGBMR = lgb.LGBMRegressor(random_state=RANDOM_STATE)\nmodel_LGBMR.fit(features_train, target_train)\nend_model_LGBMR = time.time() # время окончания\nfit_time_model_LGBMR = end_model_LGBMR - start_model_LGBMR\nprint('Время обучения model_LGBMR', fit_time_model_LGBMR)\n")


# In[138]:


get_ipython().run_cell_magic('time', '', "\nstart_model_LGBMR = time.time() # время начала\npredictions_LGBMR = model_LGBMR.predict(features_valid)\nend_model_LGBMR = time.time() # время окончания\npredict_time_model_LGBMR = end_model_LGBMR - start_model_LGBMR\nprint('Время предсказания model_LGBMR', predict_time_model_LGBMR)\n")


# In[139]:


get_ipython().run_cell_magic('time', '', '\nrmse_LGBMR = mean_squared_error(target_valid, predictions_LGBMR)**0.5\nprint(rmse_LGBMR)\n')


# In[140]:


importances(model_LGBMR, features_train)


# #### LightGBMRegressor c GridSearchCV

# Код закоментировала, так как подбор параметров происходит более часа, успешно завершился 1 раз, далее платформа не справлялась.
# Результат rmse 1361.62. (Далее в Colab проверила работу). Время обучения модели  отличается от платформы яндекс.

# In[141]:


# поиск гиперпараметров

#hyperparams = [{'n_estimators':[100, 500, 1000],  
                #'num_leaves': [n for n in range(30, 300, 30)]}]


# In[142]:


#display(tuning_model_LGBMR_GCV.best_score_*-1)


# In[143]:


#%%time

#start_model_LGBMR_GCV = time.time() # время начала
# обучение LightGBM
#model_LGBMR_GCV = lgb.LGBMRegressor(random_state=RANDOM_STATE,
                          #n_estimators=500,
                          #num_leaves=120))
#model_LGBMR_GCV.fit(features_train, target_train)
#end_model_LGBMR_GCV = time.time() # время окончания
#fit_time_model_LGBMR_GCV = end_model_LGBMR_GCV - start_model_LGBMR_GCV
#print('Время обучения model_LGBMR_GCV', fit_time_model_LGBMR_GCV)


# In[144]:


#%%time

#start_model_LGBMR_GCV = time.time() # время начала
# прогноз 
#predictions_LGBMR_GCV = model_LGBMR_GCV.predict(features_valid)
#rmse_LGBMR_GCV = mean_squared_error(target_valid, predictions_LGBMR_GCV)**0.5
#print(rmse_LGBMR_GCV)
#end_model_LGBMR_GCV = time.time() # время окончания
#predict_time_model_LGBMR_GCV = end_model_LGBMR_GCV - start_model_LGBMR_GCV
#print('Время предсказания model_LGBMR_GCV', predict_time_model_LGBMR_GCV)


# In[145]:


#importances(model_LGBMR_GCV, features_train)


# ## Анализ моделей

# In[146]:


index = ['LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor', 
         'RandomForestRegressor с RandomizedSearchCV', 'CatBoostRegressor', 
         'LightGBMRegressor без подбора параметров']


# In[147]:


data = {'RMSE':[rmse_LR,
                rmse_DTR,
                rmse_RFR,
                rmse_RFR_SearchCV,
                rmse_CBR,
                rmse_LGBMR], 
        
        'fit_time':[fit_time_model_LR,
                   fit_time_model_DTR,
                   fit_time_model_RFR,
                   fit_time_model_RFR_SearchCV,
                   fit_time_model_CBR,
                   fit_time_model_LGBMR],
        
        'predict_time':[predict_time_model_LR,
                       predict_time_model_DTR,
                       predict_time_model_RFR, 
                       predict_time_model_RFR_SearchCV,
                       predict_time_model_CBR,
                       predict_time_model_LGBMR]
       }


# In[148]:


d_model = pd.DataFrame(data=data, index=index)
d_model


# Лучшая модель с лучшим rmse оказалась модель CatBoostRegressor: лучшая rmse, время обучения разы меньше,
# чем у LightGBMRegressor без подбора параметров. По метрике rmse хорошие результаты показали RandomForestRegressor, RandomForestRegressor с RandomizedSearchCV, CatBoostRegressor и LightGBMRegressor без подбора параметров. Худшая модель LinearRegression - не прошла порог rmse = 2500.

# In[149]:


# поверка лучшей модели на тестовой выборке


# In[150]:


#%%time

# обучение
model_CBR = CatBoostRegressor(loss_function='RMSE')
model_CBR.fit(features_train, target_train, cat_features, verbose=10)


# In[151]:


get_ipython().run_cell_magic('time', '', '\n# прогноз\nprobabilities_test_CBR = model_CBR.predict(features_test)\n')


# In[152]:


get_ipython().run_cell_magic('time', '', '\nrmse_CBR = mean_squared_error(target_test, probabilities_test_CBR)**0.5\nprint()\nprint(rmse_CBR)\n')


# ## Вывод

# Были предоставлены исходные данные в размере 354 369 строк с индексацией 0-354368 и 16 столбцов. Типы столбцов корректные, названия столбцов и названия брендов автомобилей проиведены к корректному виду (приведены к нижнему регистру, удалены пробелы в начале и в конце наименования).<br> В результате предобработки данных были удалены 4 явных дубликата и неинформатиные столбцы ( столбцы с датами: 'date_crawled' (дата скачивания анкеты из базы), 'registration_month' (месяц регистрации автомобиля), 'date_created' (дата создания анкеты), 'last_seen' (дата последней активности пользователя); столбец 'number_of_pictures' (количество фотографий автомобиля) имеет нулевые данные и, соответственно, не информативен; 'postal_code' (почтовый индекс владельца) не влияет на цену)<br>Удалены выбросы в столбцах 'price', 'registration_year', 'power', 'kilometr'.<br> В категориальных переменнтых оставшиеся пропуски заменены на 'unknown'. В итоге количество строк сократилось на с 354369 до 232332, а количество столбцов на с 16 до 10.<br>
# В результате исследования выявлены топовые автомобили по брендам, по типам кузова, по типу топлива и по типу коробки передач.<br>Для дальнейшего обучения моделей использовали кодировку категориальных признаков. Для преобразования данных - методику порядкового кодирования Original Encoding для работы с моделями, основанными на деревьях - дерево решений, случайный лес. Для линейных моделей используем прямое кодирование One Hot Encoding. Для остальных моделей - незакодированные данные.

# По важности признаков во всех моделях, кроме LightGBMRegressor без подбора параметров, лидировали признаки: <br> -год регистрации автомобиля, <br> -мощность, <br> -пробег, <br> -марка и бренд. <br>У LightGBMRegressor без подбора параметров: <br>-марка <br> -год регистрации автомобиля, <br> -мощность, <br> -пробег, <br> -бренд. 

# Лучшая модель с лучшим rmse оказалась модель CatBoostRegressor: лучшая rmse, время обучения разы меньше,
# чем у LightGBMRegressor без подбора параметров. По метрике rmse хорошие результаты показали RandomForestRegressor, RandomForestRegressor с RandomizedSearchCV, CatBoostRegressor и LightGBMRegressor без подбора параметров. Худшая модель LinearRegression- не прошла порог rmse = 2500.<br> (Если брать в расчет закомментированную модель LightGBMRegressor c GridSearchCV, то она оказалась лучшей с rmse 1361.62).
