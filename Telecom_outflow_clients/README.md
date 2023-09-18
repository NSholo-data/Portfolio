README.md

# ПРОГНОЗИРОВАНИЕ ОТТОКА КЛИЕНТОВ ОПЕРАТОРА СВЯЗИ
[ipynb](https://github.com/NSholo-data/Portfolio/blob/main/Telecom_outflow_clients/telecom_outflow_clients.ipynb)


### Описание проекта
Исследовательский анализ данных и создание модели для прогнозирования оттока клиентов на основании персональных данных о некоторых клиентах, информацию об их тарифах и договорах.

### Инструменты и навыки
pandas, <br>
numpy, <br>
seaborn, <br>
matplotlib,<br> 
phik, <br>
sklearn.model_selection import train_test_split, GridSearchCV, <br>
sklearn.pipeline.Pipeline, <br>
sklearn.pipeline import make_pipeline, <br>
sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, <br>
sklearn.compose import make_column_transformer, <br>
sklearn.linear_model import LogisticRegression, <br>
sklearn.ensemble import RandomForestClassifier, <br>
sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, <br>
catboost, <br>
предобработка данных, анализ данных, визуализация, подбор гиперпараметров, кросс-валидация

### Общий вывод
Проведены: предобработка исследовательский анализ данных. Подготовлены признаки для обучения моделей. На основе выбранных и добавленных дополнительных признаках обучено три модели. Выбор модели и подбор гиперпараметров осуществлялся с помощью GridSearchCV с кросс-валидацией (cv=5). Лучший результат для прогнозирования оттока показала модель CatBoostClassifier.
