README.md

# Восстановление золота из руды
[ipynb](https://github.com/NSholo-data/Portfolio/blob/main/Gold%20recovery/gold_recovery.ipynb)


### Описание проекта
На основании данных с параметрами добычи и очистки подготовить прототип модели машинного обучения. Компания разрабатывает решения для эффективной работы промышленных предприятий. Требуется подготовить прототип модели машинного обучения, которая должна предсказать коэффициент восстановления золота из золотосодержащей руды. Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.

### Навыки и инструменты
python, <br>
pandas, <br>
numpy, <br> 
seaborn, <br>
matploylib, <br>
plotly,<br>
from sklearn.metrics.make_scorer, mean_squared_error<br>
sklearn.preprocessing.StandardScaler, OneHotEncoder<br>
from sklearn.compose.ColumnTransformer<br>
from sklearn.model_selection.cross_val_score<br>
from sklearn.model_selection.GridSearchCV<br>
from sklearn.tree.DecisionTreeRegressor<br>
from sklearn.ensemble.RandomForestRegressor<br>
from sklearn.linear_model.LinearRegression<br>
from numpy.random.RandomState<br>
from sklearn.dummy.DummyRegressor<br>

## Общий вывод
Проведен анализ данных. Обучено 3 модели для стадий грубой и тонкой очистки, проведена проверка лучшей модели на тестовой выборке и выбрана одна для запуска в производство.
