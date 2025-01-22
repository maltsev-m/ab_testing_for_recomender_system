import os
from typing import List
from datetime import datetime

import pandas as pd
import hashlib
import uvicorn
from loguru import logger
from catboost import CatBoostClassifier
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine

from database import SQLALCHEMY_DATABASE_URL
from schema import PostGet, Response


app = FastAPI()

SALT = 'mma_salt'
GROUP_COUNT = 2

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def get_model_path(model_version: str) -> str:
    MODEL_PATH = f'service/model_{model_version}'
    return MODEL_PATH

def load_models(model_version: str):
    # Загрузка модели
    logger.info(f'loading model_{model_version}')

    model_path = get_model_path(model_version)

    model = CatBoostClassifier()

    return model.load_model(model_path)
    
def load_features():
    # Загрузка уникальных постов и юзеровов с лайками
    logger.info('loading liked posts')

    user_post_like_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        WHERE action = 'like'
    """
    liked_posts = batch_load_sql(user_post_like_query)

    # Загрузка фичей по юзерам и постам
    logger.info('loading users features')
    users_features = batch_load_sql('maksim_maltsev_users_lesson_22')
    logger.info('loading posts_test features')
    posts_features_control = batch_load_sql('maksim_maltsev_posts_lesson_22')
    logger.info('loading posts_control features')
    posts_features_test = batch_load_sql('maksim_maltsev_posts_lesson_10')

    return [liked_posts, posts_features_test, users_features, posts_features_control]

def get_exp_group(user_id: int) -> str:
    value_str = str(user_id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16) % GROUP_COUNT
    if value_num == 0:
        return "control"
    else:
        return "test"
    
def get_model_test(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    # фильтруем пользователя по id
    logger.info(f'loading user_{id} features')
    user_features = features[2].loc[features[2].user_id == id]
    user_features.drop('user_id', axis=1, inplace=True)

    # Загрузим фичи постов для юзера
    logger.info(f'loading post features for user_{id}')
    post_features = features[1].drop('text', axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    # Объединим фичи
    logger.info('concating features')
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_id_df = post_features.assign(**add_user_features)
    user_id_df = user_id_df.set_index('post_id')

    # Добавим фичи из даты
    logger.info('adding date features')
    user_id_df['hour'] = time.hour
    user_id_df['month'] = time.month

    # Определим вероятности для постов
    logger.info(f'predicting posts for user_{id}')
    logger.info(f'using model_test')
    predicts = model_test.predict_proba(user_id_df)[:,1]
    user_id_df['predicts'] = predicts
    
    # Уберем лайкнутые посты
    logger.info(f'deliting liked posts by user_{id}')
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    user_id_predict = user_id_df[~user_id_df.index.isin(liked_posts)]

    # Рекомендуем топ постов
    post_predict = user_id_predict.sort_values('predicts', ascending=False)[:limit].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in post_predict
    ]
    
def get_model_control(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    # фильтруем пользователя по id
    logger.info(f'loading user_{id} features')
    user_features = features[2].loc[features[2].user_id == id]
    user_features.drop('user_id', axis=1, inplace=True)

    # Загрузим фичи постов для юзера
    logger.info(f'loading post features for user_{id}')
    post_features = features[3].drop('text', axis=1)
    content = features[3][['post_id', 'text', 'topic']]

    # Объединим фичи
    logger.info('concating features')
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_id_df = post_features.assign(**add_user_features)
    user_id_df = user_id_df.set_index('post_id')

    # Добавим фичи из даты
    logger.info('adding date features')
    user_id_df['hour'] = time.hour
    user_id_df['month'] = time.month

    # Определим вероятности для постов
    logger.info(f'predicting posts for user_{id}')
    logger.info(f'using model_control')
    predicts = model_control.predict_proba(user_id_df)[:,1]
    user_id_df['predicts'] = predicts
    
    # Уберем лайкнутые посты
    logger.info(f'deliting liked posts by user_{id}')
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    user_id_predict = user_id_df[~user_id_df.index.isin(liked_posts)]

    # Рекомендуем топ постов
    post_predict = user_id_predict.sort_values('predicts', ascending=False)[:limit].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in post_predict
    ]

# Положим модель и фичи в соответствующие переменные при поднятии сервиса
features = load_features()
model_test = load_models('test')
model_control = load_models('control')
logger.info('service is up and running')
    
@app.get("/post/recommendations/", response_model=Response)
def get_post_recom(id: int, time: datetime, limit: int = 10) -> Response:
    # определяем группу пользователя и соответствующую модель
    logger.info(f'calculating user_{id} exp_group')
    exp_group = get_exp_group(id)

    if exp_group == "control":
        result = get_model_control(id, time, limit)
    elif exp_group == "test":
        result = get_model_test(id, time, limit)
    else:
        logger.info(f'unknown group user_{id} exp_group')
        raise ValueError('unknown group')
    
    logger.info(f'user_{id} exp_group: {exp_group}')

    return Response(
        exp_group = exp_group,
        recommendations = result
    )

if __name__ == '__main__':
    uvicorn.run(app)