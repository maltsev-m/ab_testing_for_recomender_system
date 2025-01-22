## Возможность А/Б-тестирования для рекомендательной системы.
Добавить в рекомендательный сервис возможность проведения А/Б-тестирования

Работа выполнена на Python 3.9.19.

### Стек:
PyTorch, loguru, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, CatBoost, LightGBM, FastAPI, SQLAlchemy, Pydantic

### Задачи: 
1. Обучить вторую модель, используя для обработки текстов постов энкодер от трансформера при помощи библиотеки PyTorch. Данную модель принять за тестовую, модель из 2-го этапа — за контрольную;
2. Добавить разделение пользователей на 2 группы при помощи хэширования.
3. Добавить в сервис возможность применять одну из двух моделей для рекомендаций в зависимости от группы пользователя, логировать, какая модель применялась
4. В ответе endpoint-а указать группу, в которую попал пользователь ("control" или "test")

### Полученные результаты:
endpoint работает корректно, для разных групп пользователей применяются разные модели, примененные модели отображаются в логах

### Рабочие файлы:
**control_model.ipynb** - ноутбук с обработкой признаков и обучением модели. Тексты постов обработаны TF-IDF векторизатором с предварительной лемматизацией.

**test_model.ipynb** - ноутбук с обработкой признаков и обучением модели. Тексты постов векторизированы энкодером от трансформера Distil Bert при помощи библиотеки PyTorch

**service** -  папка с файлами сервиса

**endpoint.py** - файл приложения с endpoint.

**model_control** - предобученная контрольная модель CatBoost

**model_test** - предобученная тестовая модель CatBoost

**table_user.py** - ORM таблица пользователей

**table_post.py** - ORM таблица постов социальной сети

**table_feed.py** - ORM таблица взаимодействий пользователей и постов

**database.py** - скрипт с подключение к базе данных

**schemas.py** - модели валидации pydentic

**requirements.txt** - необходимые библиотеки