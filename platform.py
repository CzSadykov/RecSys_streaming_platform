import os
import pickle
import sys
from typing import List
from dotenv import load_dotenv

import implicit
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy import sparse
from implicit.als import AlternatingLeastSquares

load_dotenv()

app = FastAPI()


class User(BaseModel):
    """Class of json output"""
    user_id: int
    personal: List


def process_data(path_from: str):
    """Function process

    Parameters
    ----------
    path_from : str
        path to read data

    Returns
    -------
    data: pandas.DataFrame
        dataframe after proccessing
    sparse_item_user: scipy.sparse.csc_matrix
        sparce item user csc matrix
    """
    cols = [
        'uid', 'session_id', 'streamer_name', 'time_start', 'time_end'
        ]
    data = pd.read_csv(path_from, header=None, names=cols)
    data["uid"] = data["uid"].astype("category")
    data["streamer_name"] = data["streamer_name"].astype("category")

    data["user_id"] = data["uid"].cat.codes
    data["streamer_id"] = data["streamer_name"].cat.codes
    data['total_time_stream'] = data['time_end'] - data['time_start']

    sparse_item_user = sparse.csr_matrix(
        (data["total_time_stream"],
         (data["streamer_id"], data["user_id"])),
        shape=(data["streamer_id"].nunique(), data["user_id"].nunique())
        )
    return data, sparse_item_user


def fit_model(
    sparse_item_user,
    model_path: str,
    iterations: int = 12,
    factors: int = 500,
    regularization: float = 0.2,
    alpha: float = 100,
    random_state: int = 42,
):
    """function fit ALS

    Parameters
    ----------
    sparse_item_user : csr_matrix
        Сompressed Sparse Row matrix
    model_path: str
        Path to save model as pickle format
    iterations : int, optional
        Number of iterations, by default 12
    factors : int, optional
        Number of factors, by default 500
    regularization : float, optional
        Regularization, by default 0.2
    alpha : int, optional
        Alpha increments matrix values, by default 100
    random_state : int, optional
        Random state, by default 42

    Returns
    -------
    model: AlternatingLeastSquares
        trained model
    """
    als_model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        random_state=random_state
    )
    als_model.fit(
        (sparse_item_user * alpha).astype('double'), show_progress=True
        )

    try:
        with open(model_path, 'wb') as f:
            pickle.dump(als_model, f)
    except Exception:
        print(f"Error: model not saved in {model_path}")
        raise Exception("Model not saved")

    return als_model


def load_model(
    model_path: str,
):
    """Function that loads model from path

    Parameters
    ----------
    path : str
        Path to read model as pickle format

    Returns
    -------
    model: AlternatingLeastSquares
        Trained model
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def personal_recommendations(
    user_id: int,
    n_similar: int,
    model: implicit.als.AlternatingLeastSquares,
    data: pd.DataFrame,
    sparse_item_user: sparse.csr_matrix,
) -> List:
    """Give similar items from model

    Parameters
    ----------
    user_id : int
        User to whom we will recommend similar items
    n_similar : int
        Number of similar items
    model : als.AlternatingLeastSquares
        ALS model
    data : pd.DataFrame
        DataFrame containing streamer names & their ids

    Returns
    -------
    similar_items: List
        list of similar items to recommend for a user
    """
    if user_id not in data['uid'].unique():
        return []

    internal_user_id = data[data['uid'] == user_id]['user_id'].iloc[0]
    recommended = model.recommend(
        internal_user_id, sparse_item_user[internal_user_id], N=n_similar
        )

    if not recommended:
        return []

    similar_items = []

    for i in recommended:
        streamer_id = i[0]
        streamer_data = data[data.streamer_id == streamer_id]
        if not streamer_data.empty:
            streamer_name = streamer_data['streamer_name'].values[0]
            similar_items.append(streamer_name)

    return similar_items


@app.get("/recomendations/user/{user_id}")
async def get_recommendation(user_id: int):
    """Fast Api Web Application

    Parameters
    ----------
    user_id : int
        user to whom we will recommend streamers

    Returns
    -------
    user: json
        user informations
    """
    data_path = os.path.join(sys.path[0], os.getenv("data_path"))
    model_path = os.path.join(sys.path[0], os.getenv("model_path"))
    model = load_model(model_path)
    data, sparse_item_user = process_data(data_path)

    if user_id not in data['uid'].unique():
        raise HTTPException(status_code=404, detail="User not found")

    personal = personal_recommendations(
        user_id, 100, model, data, sparse_item_user
        )
    user = User(user_id=user_id, personal=personal)
    return user


def main() -> None:
    """Run application"""
    uvicorn.run("platform:app", host="localhost")


if __name__ == "__main__":
    main()
