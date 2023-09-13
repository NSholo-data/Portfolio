# my_projects_for_YandexPracticum

| Название проекта | Тема  | Сфера деятельности | Навыки, инструменты | Задачи проект, описание                 | Ключевые слова проекта |
|------------------|-------|--------------------|---------------------|-----------------------------------------|------------------------|
|[Продажа квартир в Санкт-Петпербурге - анализ рынка недвижимости](https://github.com/NSholo-data/my_projects_for_YandexPracticum/tree/main/research_of_ads_for_the_sale_of_apartments)|Исследовательский анализ данных|`Интернет-сервис`, `Площадки объявлений`, `Недвижимость`|`python`, `pandas`, `seaborn`, `matplotlib`, `исследовательский анализ`, `визуализация`, `предобработка данных`|Провести исследовательский анализ данных, который поможет установить параметры, влияющие на цену объектов. Это позволит построить автоматизированную систему: она отследит аномалии и мошенническую деятельность. <br><br> На основании архивных данных объявлений о продаже квартир в Санкт-Петербурге и соседних населённых пунктах за несколько лет сервиса Яндекс Недвижимость нужно научиться определять рыночную стоимость объектов недвижимости. Необходимо провести исследовательский анализ данных, который поможет установить параметры, влияющие на цену объектов. Это позволит построить автоматизированную систему: она отследит аномалии и мошенническую деятельность. По каждой квартире на продажу доступны два вида данных. Первые вписаны пользователем, вторые — получены автоматически на основе картографических данных. Например, расстояние до центра, аэропорта и других объектов — эти данные автоматически получены из геосервисов. Количество парков и водоёмов также заполняется без участия пользователя.|обработка данных, histogram, boxplot, scttermatrix, scatterplot, категоризация, мониторинг|
|[Прогнозирование оттока клиентов оператора связи](https://github.com/NSholo-data/my_projects_for_YandexPracticum/tree/main/telecom_final)|Финальный проект|`Телеком`|`phik`, `scikit-learn`, `catboost`, `pandas`, `numpy`, `seaborn`, `matplotlib`, `предобработка данных`, `анализ данных`, `визуализация`, `подбор гиперпараметров`, `кросс-валидация`|Перед нами задача бинарной классифкации. Цель - создание модели для прогнозирования оттока клиентов и достижение минимального порога AUC-ROC=0.85<br><br>Оператор связи «Ниединогоразрыва.ком» хочет научиться прогнозировать отток клиентов. Если выяснится, что пользователь планирует уйти, ему будут предложены промокоды и специальные условия. Команда оператора собрала персональные данные о некоторых клиентах, информацию об их тарифах и договорах.|подбор гиперпараметров, обработка данных, анализ данных, boxplot, histogram, группировка, сортировка|
|[Изучение закономерностей, определяющих успешность игры](https://github.com/NSholo-data/my_projects_for_YandexPracticum/tree/main/games)|Сборный проект|`Gamedev`, `оффлайн`, `Интернет-магазин`|`python`, `pandas`, `seaborn`, `matplotlib`, `numpy`, `визуализация`, `исследовательский анализ`, `описательная статистика`|Выявить определяющие успешность игры закономерности. Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании.<br><br>Интернет-магазин "Стримчик", продает по всему миру комьюторные игры. Из открытых источников доступны исторические данные о продажахигр, оценки пользователей и экспертов, жанры и платформы (например, Xbox или PlayStation). Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании. Перед нами данные до 2016 года. Мы планируем кампанию на 2017 год. В наборе данных попадается аббревиатура ESRB (Entertainment Software Rating Board) — это ассоциация, определяющая возрастной рейтинг компьютерных игр. ESRB оценивает игровой контент и присваивает ему подходящую возрастную категорию, например, «Для взрослых», «Для детей младшего возраста» или «Для подростков».|обработка данных, исследовательский анализ, статистический анализ, boxplot|
|[Исследование надежности заемщиков](https://github.com/NSholo-data/my_projects_for_YandexPracticum/tree/main/bank_borrower_reliability_research)|Предобработка данных|`Банковская сфера`, `Кредитование`|`предобработка данных`, `Pandas`, `Rython`, `группировка данных`|Нужно разобраться, влияет ли семейное положение и количество детей клиента на факт погашения кредита в срок. <br><br> Входные данные от банка — статистика о платёжеспособности клиентов.Заказчик — кредитный отдел банка. Результаты исследования будут учтены при построении модели кредитного скоринга — специальной системы, которая оценивает способность потенциального заёмщика вернуть кредит банку.|обработка данных, дубликаты, группировка, категоризация, пропуски|
|["Яндекс-Музыка" - сравнение пользователей двух городов](https://github.com/NSholo-data/my_projects_for_YandexPracticum/tree/main/yandex_music)|Базовый Python|`Интернет-сервис` `Cтриминговый сервис`|`Python`, `Pandas`|На реальных данных Яндекс.Музыки с помощью библиотеки Pandas проверить данные и сравнить поведение и предпочтения пользователей двух городов - Москвы и Санкт-Петербурга. <br><br> На данных Яндекс Музыки проверены гипотезы о поведени и предпочтениях пользователей двух столиц. Гипотезы: <br>- Активность пользователей зависит от дня недели. Причём в Москве и Петербурге это проявляется по-разному.<br> - Утром в понедельник в Москве преобладают одни жанры музыки, а в Петербурге — другие. Это верно и для вечера пятницы.<br> - Москва и Петербург предпочитают разные жанры музыки. В Москве чаще слушают поп-музыку, в Петербурге — русский рэп.|обработка данных, дубликаты, пропуски, группировка, сортировка, логическая индексация|
|[Прогноз оттока клиента банка](https://github.com/NSholo-data/my_projects_for_YandexPracticum/tree/main/bank%20-%20customer%20outflow)|Обучение с учителем|`Бизнес`, `Инвестиции`, `Банковская сфера`, `Кредитование`|`Pandas`, `Matplotlib`, `Seaborn`, `Scikit-learn`|На основании предоставленых исторических данных о поведении клиентов и расторжении договоров с банком построить модель с предельно большим значением F1-меры (довести метрику до 0.59). Проверьте F1-меру на тестовой выборке. Дополнительно измерить AUC-ROC, сравнить её значение с F1-мерой.<br><br> Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых. Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет.|классификация, бинарная классификация, подбор гиперпараметров, визуализация, выбор модели МО|
|[Защита данных клиента страховой компании](https://github.com/NSholo-data/my_projects_for_YandexPracticum/tree/main/linear_algebra)|Линейная алгебра|`Банковская сфера`, `Интернет-сервисы`, `Инвестиции`, `Телеком`|`Pandas`, `numpy`, `seabrn`, `matplotlib`, `plotly`, `scikit-learn`|Разработать такой метод преобразования данных, чтобы по ним было сложно восстановить персональную информацию. Обосновать корректность его работы.<br><br>Нужно защитить данные клиентов страховой компании «Хоть потоп». Нужно защитить данные, чтобы при преобразовании качество моделей машинного обучения не ухудшилось. Подбирать наилучшую модель не требуется.|Pandas, numpy, seabrn, matplotlib, plotly, scikit-learn, визуализация, машинное обучение, линейная алгебра, регрессия, разработка модели анонимизации персональных данных|
