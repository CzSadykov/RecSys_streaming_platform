import os
import sys
import time

from dotenv import load_dotenv
from live_streaming_platform.platform import process_data, fit_model

# Загрузка переменных окружения из файла .env
load_dotenv()

# Получение путей из переменных окружения
data_path = os.path.join(sys.path[0], os.getenv("data_path"))
model_path = os.path.join(sys.path[0], os.getenv("model_path"))

# Создание директории для моделей, если она не существует
os.makedirs(os.path.dirname(model_path), exist_ok=True)

print(f"Data path: {data_path}")
print(f"Model path: {model_path}")

# Добавление отладочной информации
print("Processing data...")
start_time = time.time()

# Изменение: передаем data_path напрямую в process_data
data, sparse_item_user = process_data(data_path)

print(f"Processing data completed in {time.time() - start_time:.2f} seconds")
print(
    f"Data size: {data.shape}, "
    f"Sparse matrix size: {sparse_item_user.shape}"
    )

print("Start training model...")
start_time = time.time()

model = fit_model(sparse_item_user, model_path)

print(f"Training model completed in {time.time() - start_time:.2f} seconds")

# Проверка, что модель сохранена
if os.path.exists(model_path):
    print(f"Model successfully saved in {model_path}")
else:
    print(f"Error: model not saved in {model_path}")

print("Process completed")
