README.md

# Восстановление золота из руды
[ipynb](https://github.com/NSholo-data/Portfolio/blob/main/Gold%20recovery/gold_recovery.ipynb)


### Описание проекта
На основании данных с параметрами добычи и очистки подготовить прототип модели машинного обучения. Компания разрабатывает решения для эффективной работы промышленных предприятий. Требуется подготовить прототип модели машинного обучения, которая должна предсказать коэффициент восстановления золота из золотосодержащей руды. Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.

### Навыки и инструменты
python, 
pandas, 
numpy, 
seaborn,
matploylib, 
plotly,
from sklearn.metrics.make_scorer, mean_squared_error
sklearn.preprocessing.StandardScaler, OneHotEncoder
from sklearn.compose.ColumnTransformer
from sklearn.model_selection.cross_val_score
from sklearn.model_selection.GridSearchCV
from sklearn.tree.DecisionTreeRegressor
from sklearn.ensemble.RandomForestRegressor
from sklearn.linear_model.LinearRegression
from numpy.random.RandomState
from sklearn.dummy.DummyRegressor

## Общий вывод
Проведен анализ данных. Обучено 3 модели для стадий грубой и тонкой очистки, проведена проверка лучшей модели на тестовой выборке и выбрана одна для запуска в производство.
