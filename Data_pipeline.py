#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle

# Define the DAG
dag = DAG(
    'stock_data_pipeline',
    description = 'Data pipeline for processing stock data and training an ML model',
    start_date = datetime(2023, 5, 17),
    schedule_interval = '0 0 * * 0',
    email = 'naveenaveeravarapu3@gmail.com'
    email_on_failure = True
    email_on_retry = True
    retries = 1
    depends_on_past = False
)

def preprocess_stock_data():
    # Load the dataset
    # Path to the folder containing the CSV files
    folder_paths = ['etfs/','stocks/']
    updated_dfs = []

    # Looping through the folders of etfs and stocks
    for folder_path in folder_paths:
        # Creating an empty list to store individual DataFrames
        dfs = []
        file_names = []

        # Iterating over each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                # Getting the full file path which includes the path and the filename
                file_path = os.path.join(folder_path, filename)
                # Reading the CSV file into a DataFrame. Since each file has it's own csv file
                df = pd.read_csv(file_path)
                # getting the filename of each csv file and loading the same in a column called symbol
                df['Symbol'] = filename.replace('.csv', '')
                # Creating different dataframes and appending that to a list
                dfs.append(df)
                # Store the filename of all the files in a list
                file_names.append(filename)

        # Combine all DataFrames in the list into a single DataFrame
        df = pd.concat(dfs, ignore_index=True)


        # Read the CSV file which has the security name. Here the file_names is being used to get the security_names
        comparison_file_path = 'symbols_valid_meta.csv'
        comparison_df = pd.read_csv(comparison_file_path)

        # Creating a new DataFrame with only the 'Symbol' and 'security_name' columns from the comparison DataFrame
        comparison_subset = comparison_df[['Symbol', 'Security Name']]

        # Merging the comparison DataFrame subset with the combined DataFrame based on the 'Symbol' column
        df = pd.merge(df, comparison_subset, on='Symbol', how='left')
        updated_dfs.append(df)


    etfs_df = updated_dfs[0]
    stocks_df = updated_dfs[1]


    combined_df = pd.concat([etfs_df, stocks_df], ignore_index=True)

    # Defining the columns so that they will be in the order as defined in the problem.
    columns = ['Symbol','Security Name', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    combined_df = combined_df[columns]

    # Convert the 'Date' column to datetime
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    # Setting the Date column as index
    combined_df.set_index('Date', inplace=True)
    
    # Calculate the moving average of the trading volume (Volume) for each stock and ETF
    grouped_df = combined_df.groupby('Symbol')
    moving_avg = grouped_df['Volume'].rolling(window=30).mean()

    # Reset the index and drop the original index to align with the combined_df
    moving_avg = moving_avg.reset_index(level=0, drop=True)

    # Assign the calculated moving average values to the vol_moving_avg column in combined_df
    combined_df['vol_moving_avg'] = moving_avg
    
    
    # Calculate the rolling median of the 'Adj Close' column for each stock and ETF
    grouped_df = combined_df.groupby('Symbol')
    rolling_median = grouped_df['Adj Close'].rolling(window=30).median()

    # Reset the index and drop the original index to align with the combined_df
    rolling_median = rolling_median.reset_index(level=0, drop=True)

    # Assign the calculated rolling median values to the adj_close_rolling_med column in combined_df
    combined_df['adj_close_rolling_med'] = rolling_median

    # Save the processed data to a new file
    combined_df.to_csv('combined_df.csv', index=False)


def train_ml_model():
    # Load the processed data
    df = pd.read_csv('combined_df.csv')
    
    # Dropping rows with missing values
    df.dropna(inplace=True)

    # Splitting the data into features and target
    features = combined_df[['vol_moving_avg', 'adj_close_rolling_med']]
    target = combined_df['Volume']

    # Step 4: splitting the features and target into Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features.values, target.values, test_size=0.2, random_state=42)

    # Setting up the logging configuration
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Training the Machine Learning model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # Logging the training metrics
    logging.info('Training Metrics:')
    # Using model that trained above for prediction 
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    logging.info(f'Mean Squared Error (MSE): {mse}')

    # Evaluating the model on the test_data
    logging.info('Evaluation Metrics:')
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f'Mean Squared Error (MSE): {mse}')

    # Saving the trained model in the pkl format
    model_filename = 'xgb_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)



# Define the tasks
preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_stock_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_ml_model,
    dag=dag,
)

# Set the task dependencies
preprocess_data_task >> train_model_task

# Specify the order of task execution
preprocess_data_task

# Run the DAG
dag


# ## Here is a more detailed description of how data flows through the pipeline and the dependencies between tasks in the provided Airflow DAG:
# 
# 1) The DAG, named 'stock_data_pipeline', is scheduled to run daily starting from May 17, 2023, as specified by the "start_date" and "schedule_interval parameters".
# 
# 2) The pipeline consists of two tasks: 'preprocess_data_task' and 'train_model_task'.
# 
# 3) The "preprocess_data_task" is the first task in the pipeline. It is defined as a PythonOperator, which means it will execute the "preprocess_stock_data" function when triggered.
# 
# 4) Inside the "preprocess_stock_data" function, the stock data is loaded from the file '/path/to/stock_data.csv' using the pandas library. The 'Date' column is converted to a datetime format.
# 
# 5) The function then calculates the moving average of the volume ('Volume') for each stock and ETF by grouping the data by the 'Symbol' column and applying the rolling window calculation. The results are stored in a new column named 'vol_moving_avg'.
# 
# 6) Similarly, the function calculates the rolling median of the adjusted close ('Adj Close') for each stock and ETF and stores the values in a new column named 'adj_close_rolling_med'.
# 
# 7) Finally, the processed data is saved to a new file '/path/to/processed_stock_data.csv' using the to_csv method.
# 
# 8) The "train_model_task" is the second task in the pipeline. It is also defined as a PythonOperator and will execute the "train_ml_model" function.
# 
# 9) Inside the "train_ml_model" function, the processed data is loaded from the file '/path/to/processed_stock_data.csv'.
# 
# 10) The data is then split into features and target, with the features consisting of the 'vol_moving_avg' and 'adj_close_rolling_med' columns, and the target being the 'Volume' column.
# 
# 11) The data is further split into training and testing sets using the train_test_split function from sklearn.model_selection.
# 
# 12) An XGBoost regressor model is initialized, and the model is trained on the training data using the fit method.
# 
# 13) The trained model is saved to the file '/path/to/trained_model.pkl' using the pickle library.
# 
# 14) The dependencies between tasks are specified using the >> operator. In this case, "preprocess_data_task >> train_model_task" indicates that the "train_model_task" depends on the successful completion of the "preprocess_data_task". This ensures that the data preprocessing is finished before starting the model training.

# In[ ]:

