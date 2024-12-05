Домашнее задание №1 (base), Homework #1 (Base)

Что было сделано и с какими результатами:
1. Проведён простейший EDA и обработка признаков в результате данные предварительно подготовлены к обучению,
2. Построены визуализации pairplot, тепловые карты с корреляциями Пирсона. По графику прослеживается связь целевой переменной с признаками: year, km_driven, engine, и max_power. Признаки mileage, torque, и seats показывают более слабую связь,
3. Обучена модель классической линейной регрессии только на вещественных признаках и произведена её оценка R^2 и MSE где R2 для теста: 0.5945; MSE для теста: 233097066692.6146; R2 для трейна: 0.5923; MSE для трейна: 116875266859.4052,
4. Стандартизированы признаки при помощи StandardScaler,
5. Произведена оценка коэффициента βi. Наиболее информативным в предсказании цены является признак year среди вещественных признаков,
6. Обучена Lasso регрессия только на вещественных признаках. R^2 для Lasso трейна: 0.5923; MSE для Lasso трейна: 116875266870.5470; R^2 для Lasso теста: 0.5945; MSE для Lasso теста: 233097774128.8036. Ввиду малого колличества признаков и наличию корреляций с target у каждого из них Lasso не зануляет признаки,
7. Обучена Lasso регрессия только на вещественных признаках с приминением кросс-валидации. R^2 для Lasso трейна с кросс-валидацией: 0.5922; MSE для Lasso трейна с кросс-валидацией: 116886342034.7602; R^2 для Lasso теста с кросс-валидацией: 0.5932; MSE для Lasso теста с кросс-валидацией: 233812622613.3236,
8. Обучена ElasticNet регрессия только на вещественных признаках с приминением кросс-валидации. R^2 для ElasticNet трейна с кросс-валидацией: 0.5896; MSE для ElasticNet трейна с кросс-валидацией: 117630289608.1200; R^2 для ElasticNet теста с кросс-валидацией: 0.5770; MSE для ElasticNet теста с кросс-валидацией: 243129607740.0414,
9. Добавлены категориальные признаки, произведена дополнительная обработка признаков
10. Закодированы категориалльные признаки методом One-Hot Encoding,
11. Обучена модель Ridge регрессии на обработанном датафрейме с приминением кросс-валидации. Лучшее значение R2 для трейна: 0.7692926231567108; Значение R^2 для теста: 0.9104859007128742; Значение MSE для трейна: 28939334544.7721; Значение MSE для теста: 51455260402.4948,
12. Построена кастомная метрика показывающая долю прогнозов, отличающихся от реальных цен на авто не более чем на 10% для всех обученных моделей. Самый высокий показатель у Ridge модели: на тесте: 36.30%; на трейне: 35.03%,
13. Реализован веб-сервис для применения построенной модели на новых данных при помощи FastAPI, скриншоты и адрес к скринкасту ниже,
14. Записано предположение, что кот с последнего фото товарищ Sabrina Sadiekh (@sabrina_sadiekh).

Максимальный "буст" дало включение категориалльных признаков, особенно обработка столбца 'name'

Не получилось уменьшить разницу в показаниях R^2 и MSE для тестовых и тренировочных данных Ridge регрессии.

-------------------------------------------------------------------------------
What was done and Results:
1. Conducted simple EDA and feature processing, resulting in data being preliminarily prepared for training,
2. Created visualizations: pair plots and Pearson correlation heatmaps. The graphs reveal relationships between the target variable and features: year, km_driven, engine, and max_power. Features mileage, torque, and seats show weaker relationships,
3. Trained a classical linear regression model using only numerical features and evaluated it: R^2 for test: 0.5945; MSE for test: 233097066692.6146; R^2 for train: 0.5923; MSE for train: 116875266859.4052,
4. Standardized the features using StandardScaler,
5. Assessed the coefficients 𝛽𝑖. The feature year is the most informative for predicting price among numerical features,
6. Trained a Lasso regression model using only numerical features. R^2 for Lasso train: 0.5923; MSE for Lasso train: 116875266870.5470; R^2 for Lasso test: 0.5945; MSE for Lasso test: 233097774128.8036. Due to the small number of features and their correlations with 
the target, Lasso did not eliminate any features,
7. Trained a Lasso regression model using only numerical features with cross-validation. R^2 for Lasso train with cross-validation: 0.5922; MSE for Lasso train with cross-validation: 116886342034.7602; R^2 for Lasso test with cross-validation: 0.5932; MSE for Lasso test with cross-validation: 233812622613.3236,
8. Trained an ElasticNet regression model using only numerical features with cross-validation. R^2 for ElasticNet train with cross-validation: 0.5896; MSE for ElasticNet train with cross-validation: 117630289608.1200; R^2 for ElasticNet test with cross-validation: 0.5770; MSE for ElasticNet test with cross-validation: 243129607740.0414,
9. Added categorical features, performed additional feature processing,
10. Encoded categorical features using One-Hot Encoding,
11. Trained a Ridge regression model on the processed dataframe with cross-validation. Best results: R^2 for train: 0.7692926231567108; R^2 for test: 0.9104859007128742; MSE for train: 28939334544.7721; MSE for test: 51455260402.4948,
12. Built a custom metric showing the share of predictions differing from actual car prices by no more than 10% for all trained models. The highest score was achieved by the Ridge model: on test: 36.30%; on train: 35.03%,
13. Developed a web service to apply the trained model to new data using FastAPI, screenshots, and a link to the screencast provided below,
14. Recorded the hypothesis that the cat in the last photo belongs to Sabrina Sadiekh (@sabrina_sadiekh).

The inclusion of categorical features, especially processing the name column, provided the most significant boost.

A larger dataset is required to reduce the discrepancy in R^2 and MSE values between the train and test datasets for the Ridge regression.

-------------------------------------------------------------------------------
Выгруженный файл csv (1) включающий столбец selling_prise и predictions![Downloaded csv file (1) including the columns selling_prise and predictions](https://github.com/Aleksei-Ia/ML_2024/blob/1268575027471b9b8104cf56cf625eb9ff1bf4bd/Images/2024-12-04_201309.png)
Загруженный файл csv (1) включающий столбец selling_prise![Uploaded csv file (1) including the column selling_prise](https://github.com/Aleksei-Ia/ML_2024/blob/1268575027471b9b8104cf56cf625eb9ff1bf4bd/Images/2024-12-04_201333.png)
Загруженный файл csv (2) без столбца selling_prise![Uploaded csv file (2) without the column selling_price](https://github.com/Aleksei-Ia/ML_2024/blob/1268575027471b9b8104cf56cf625eb9ff1bf4bd/Images/2024-12-04_201348.png)
Выгруженный файл csv (2) включающий столбец predictions![Downloaded csv file (2) including the column predictions](https://github.com/Aleksei-Ia/ML_2024/blob/1268575027471b9b8104cf56cf625eb9ff1bf4bd/Images/2024-12-04_201404.png)
Терминал после загрузки сервиса![Terminal After Service Launch](https://github.com/Aleksei-Ia/ML_2024/blob/1268575027471b9b8104cf56cf625eb9ff1bf4bd/Images/2024-12-04_201446.png)
Работа predict_csv при помощи FastAPI-Swagger_UI в браузере![Running predict_csv with FastAPI via Swagger UI in browser](https://github.com/Aleksei-Ia/ML_2024/blob/1268575027471b9b8104cf56cf625eb9ff1bf4bd/Images/2024-12-04_201557.png)
Работа predict_item при помощи FastAPI-Swagger_UI в браузере![Running predict_item with FastAPI via Swagger UI in browser](https://github.com/Aleksei-Ia/ML_2024/blob/1268575027471b9b8104cf56cf625eb9ff1bf4bd/Images/2024-12-04_201917.png)

Ссылка к скринкасту, screencast link
[Google Drive](https://drive.google.com/file/d/1Tmamng4XC53j_2MLMmQsZ2vTi0krn4Fp/view?usp=sharing)
