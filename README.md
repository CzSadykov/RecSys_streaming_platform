# Live Streaming Platform Recommendation System

This project implements a recommendation system for a live streaming platform using the Implicit Alternating Least Squares (ALS) algorithm. The system provides personalized streamer recommendations to users based on their viewing history.

## Project Structure

The project consists of the following main components:

- `platform.py`: Contains the core functionality of the recommendation system and the FastAPI web application.
- `als_fit.py`: Script for processing data and training the Implicit ALS model.
- `cold_start.py`: Implements a baseline recommendation system for new users or when personalized recommendations are not available.
- `metrics.py`: Implements evaluation metrics for assessing the recommendation system's performance.
- `requirements.txt`: Lists all the required Python packages for the project.

## Data Structure and Processing

Current platform relies on a following data structure in csv format:

- `uid`: Unique identifier for each user
- `session_id`: Unique identifier for each viewing session
- `streamer_name`: Name of the streamer
- `time_start`: Start time of the viewing session
- `time_end`: End time of the viewing session

However, you can easily modify the code to work with other data structures (see `process_data` function in `platform.py`)

## Setup and Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up the environment variables by creating a `.env` file in the project root with the following content:
   ```
   data_path=path/to/your/data.csv
   model_path=path/to/save/model.pkl
   ```

## Core Functionality

### Data Processing (`process_data` function in `platform.py`)

- Reads the input CSV file
- Processes the data, converting user and streamer IDs to categorical codes
- Creates a sparse item-user matrix

### Model Training (`fit_model` function in `platform.py`)

- Trains an ALS model using the implicit library
- Saves the trained model to the specified path

### Recommendation Generation (`personal_recommendations` function in `platform.py`)

- Loads the trained model
- Generates personalized recommendations for a given user

### Baseline Recommendations (`cold_start.py`)

The `cold_start.py` script implements a baseline recommendation system for new users or when personalized recommendations are not available. Its logic includes:

- Analyzing the overall popularity of streamers across all users
- Generating recommendations based on the most-watched streamers
- Providing a fallback mechanism when the ALS model cannot generate personalized recommendations
- Potentially incorporating basic demographic or category-based filtering for slightly more targeted recommendations

This baseline approach ensures that the system can always provide recommendations, even for users with no viewing history.

### Evaluation Metrics (`metrics.py`)

The `metrics.py` file contains functions for evaluating the performance of the recommendation system. Here's an accurate description of its functionality:

1. Mean Average Precision at K (MAP@K):
   - Computes the mean of the Average Precision at K across all users.
   - Provides an overall measure of recommendation quality across the user base.

2. Normalized Discounted Cumulative Gain at K (NDCG@K):
   - Measures the ranking quality of the recommendations.
   - Takes into account the position of correct recommendations in the list.
   - Normalized version allows for comparison between users with different numbers of relevant items.

These metrics focus on evaluating the ranking quality and relevance of the recommendations, which are crucial aspects of a recommendation system's performance in a streaming platform context.

### FastAPI Web Application (`platform.py`)

The `platform.py` file is the core of the recommendation system and contains the following key components:

1. Data Processing:
   - The `process_data` function reads and processes the input CSV file, converting user and streamer IDs to categorical codes and creating a sparse item-user matrix.

2. Model Operations:
   - `fit_model`: Trains the ALS model using the implicit library and saves it to the specified path.
   - `load_model`: Loads a previously trained model from the specified path.

3. Recommendation Generation:
   - `personal_recommendations`: Generates personalized recommendations for a given user using the trained ALS model.

4. FastAPI Application:
   - Defines the FastAPI app and sets up the API endpoint.
   - The `/recomendations/user/{user_id}` endpoint handles GET requests for user recommendations.
   - It loads the model, processes the data, and returns personalized recommendations.

5. Error Handling:
   - Implements error handling for cases such as user not found in the dataset.

6. Main Function:
   - Sets up and runs the uvicorn server to host the FastAPI application.

The `platform.py` file integrates all these components to provide a complete recommendation system, from data processing to serving recommendations via an API.

## Usage

1. Train the model:
   ```
   python als_fit.py
   ```
   This script processes the data and trains the ALS model, saving it to the specified path.

2. Run the FastAPI application:
   ```
   python platform.py
   ```
   This starts the web server, making the recommendation API available.

3. Access recommendations:
   Send a GET request to `http://localhost:8000/recomendations/user/{user_id}` to receive personalized recommendations for a specific user.

4. Evaluate the model:
   Use the functions in `metrics.py` to assess the performance of the recommendation system, focusing on MAP@K and NDCG@K to evaluate ranking quality and relevance of recommendations.

## Environment Variables

The project uses two main environment variables:

- `data_path`: Path to the input CSV file containing user viewing data
- `model_path`: Path where the trained model will be saved

These variables should be set in the `.env` file in the project root directory.

## Dependencies

Main dependencies include:
- numpy
- pandas
- uvicorn
- scipy
- implicit
- fastapi
- python-dotenv

For the full list of dependencies and their versions, refer to `requirements.txt`.

## Note

Ensure that the data file is in the correct format and location as specified in the `data_path` environment variable. The system expects a CSV file with columns for user ID, session ID, streamer name, start time, and end time.
