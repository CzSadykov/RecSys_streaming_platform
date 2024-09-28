import os
import sys
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class User(BaseModel):
    user_id: int
    time: int
    popular_streamers: List


def process_data(path_from: str, time_now: int = 6147):
    """Function process

    Parameters
    ----------
    path_from : str
        path to read data
    time_now : int
        time to filter data

    Returns
    -------
    data: pandas.DataFrame
        dataframe after proccessing
    """
    cols = [
        'uid', 'session_id', 'streamer_name', 'time_start', 'time_end'
        ]
    data = pd.read_csv(path_from, header=None, names=cols)
    data = data[(data["time_start"] < time_now) & (data["time_end"] > time_now)]
    return data


def recomend_popularity(data: pd.DataFrame):
    """Recomend Popularity

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    popular_streamers: List
    """
    popular_streamers = data.groupby('streamer_name')['uid']\
        .count().sort_values(ascending=False).index.tolist()
    return popular_streamers


@app.get("/popular/user/{user_id}")
async def get_popularity(user_id: int, time: int = 6147):
    """Fast Api Web Application

    Parameters
    ----------
    user_id : int
        user id
    time : int, optional
        time, by default 6147

    Returns
    -------
    user: json
        user informations
    """
    if "data_path" not in os.environ:
        print("Внимание: переменная окружения 'data_path' не установлена.")
        os.environ["data_path"] = "data/data_recsys.csv"  # Set default path

    data_path = os.path.join(sys.path[0], os.environ["data_path"])

    data = process_data(data_path, time)
    popular_streamers = recomend_popularity(data)
    user = User(
        user_id=user_id, time=time, popular_streamers=popular_streamers
        )
    return user


def main() -> None:
    """Run application"""
    uvicorn.run("cold_start:app", host="localhost")


if __name__ == "__main__":
    main()
