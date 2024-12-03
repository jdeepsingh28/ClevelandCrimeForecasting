import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, BatchNormalization, Embedding, Flatten, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.utils.class_weight import compute_class_weight


def preprocess_crime_data(crime):
    """
    Preprocesses the raw crime data by cleaning, encoding, and feature engineering.

    Parameters:
    - crime: Raw crime DataFrame.

    Returns:
    - Processed crime DataFrame.
    """
    # Ensure 'offense_date' is in datetime format
    crime['offense_date'] = pd.to_datetime(crime['offense_date'], errors='coerce')
    crime.dropna(subset=['offense_date'], inplace=True)

    # Drop unnecessary columns
    cols_to_drop = ['census_tract_geoid', 'census_block_group', 'census_bg_geoid', 'census_block_geoid',
                    'std_parcelpin',
                    'address_public', 'object_id', 'primary_key', 'case_number', 'reported_date', 'dow_name', 'statute',
                    'stat_desc', 'date', 'days_ago', 'geoid', 'city', 'zip', 'ward', 'primary_key', 'district',
                    'time_group',
                    'census_tract', 'time_block']
    crime = crime.drop(columns=cols_to_drop, errors='ignore')

    # Handle date discrepancies
    crime['extracted_month'] = crime['offense_date'].dt.month
    crime['extracted_day'] = crime['offense_date'].dt.day
    crime = crime.drop(columns=['offense_month', 'offense_day'], errors='ignore')

    # Drop duplicates
    crime = crime.drop_duplicates()

    # Encode cyclical features
    crime['dow'] = crime['offense_date'].dt.weekday + 1  # Monday=1, Sunday=7
    crime['dow_sin'] = np.sin(2 * np.pi * (crime['dow'] - 1) / 7)
    crime['dow_cos'] = np.cos(2 * np.pi * (crime['dow'] - 1) / 7)
    crime['day_sin'] = np.sin(2 * np.pi * (crime['extracted_day'] - 1) / 31)
    crime['day_cos'] = np.cos(2 * np.pi * (crime['extracted_day'] - 1) / 31)
    crime['month_sin'] = np.sin(2 * np.pi * (crime['extracted_month'] - 1) / 12)
    crime['month_cos'] = np.cos(2 * np.pi * (crime['extracted_month'] - 1) / 12)
    crime['hour_sin'] = np.sin(2 * np.pi * crime['hour_of_day'] / 24)
    crime['hour_cos'] = np.cos(2 * np.pi * crime['hour_of_day'] / 24)

    # Drop original cyclical columns
    crime = crime.drop(columns=['dow', 'hour_of_day', 'extracted_month', 'extracted_day'], errors='ignore')

    # Label Encoding for categorical variables
    le_ucr_desc = LabelEncoder()
    crime['ucr_desc_numeric'] = le_ucr_desc.fit_transform(crime['ucr_desc'])
    le_offense_year = LabelEncoder()
    crime['offense_year_numeric'] = le_offense_year.fit_transform(crime['offense_year'])
    le_census_block = LabelEncoder()
    crime['census_block_numeric'] = le_census_block.fit_transform(crime['census_block'])

    # Save the LabelEncoder for census_block
    joblib.dump(le_census_block, 'le_census_block.pkl')

    # Drop original categorical columns
    crime = crime.drop(columns=['ucr_desc', 'offense_year', 'census_block'], errors='ignore')

    # Feature Engineering
    crime['week_of_year'] = crime['offense_date'].dt.isocalendar().week
    crime['temp_range'] = crime['temp_max'] - crime['temp_min']

    # Interaction Features
    crime['week_precipitation_interaction'] = crime['week_of_year'] * crime['precipitation_sum']
    crime['daylight_precipitation_interaction'] = crime['daylight_duration'] * crime['precipitation_sum']
    crime['block_week_interaction'] = crime['census_block_numeric'] * crime['week_of_year']
    crime['block_temp_max_interaction'] = crime['census_block_numeric'] * crime['temp_max']
    crime['temp_range_precipitation_interaction'] = crime['temp_range'] * crime['precipitation_sum']
    crime['precipitation_sum_hours_interaction'] = crime['precipitation_sum'] * crime['precipitation_hours']

    # Holiday Feature
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=crime['offense_date'].min(), end=crime['offense_date'].max())
    crime['is_holiday'] = crime['offense_date'].dt.normalize().isin(holidays).astype(int)

    return crime


def get_data(file_path):
    """
    Reads the crime and weather preprocessed data from a CSV file.
    """
    data = pd.read_csv(file_path)
    print("Columns in data:", data.columns)
    print("Number of records:", data.shape[0])
    return data


def add_weekly_lag_features(df, target_column, lags):
    """
    Adds lag features for weekly grouped data for the specified target column at the given lag intervals.
    """
    df = df.sort_values(['census_block_numeric', 'date'])
    for lag in lags:
        lag_column_name = f"{target_column}_lag{lag}w"
        df[lag_column_name] = df.groupby('census_block_numeric')[target_column].shift(lag)
    return df


def add_rolling_features(df, target_column, windows):
    """
    Adds rolling mean features for the specified target column at the given window sizes.
    """
    df = df.sort_values(['census_block_numeric', 'date'])
    for window in windows:
        rolling_column_name = f"{target_column}_rolling{window}"
        df[rolling_column_name] = df.groupby('census_block_numeric')[target_column].transform(
            lambda x: x.shift(1).rolling(window=window).mean()
        )
    return df


def create_time_series(data):
    """
    Aggregates the data on a weekly basis and computes weekly crime counts.
    Adds lag features and interaction features as specified.
    """
    # Ensure 'offense_date' is datetime
    data['offense_date'] = pd.to_datetime(data['offense_date'], errors='coerce')

    # Set 'offense_date' as index for grouping
    data.set_index('offense_date', inplace=True)

    data = data.sort_values(by=['census_block_numeric', 'offense_date'])

    # Define the aggregation functions for each feature
    agg_dict = {
        'dow_sin': 'mean',
        'dow_cos': 'mean',
        'day_sin': 'mean',
        'day_cos': 'mean',
        'month_sin': 'mean',
        'month_cos': 'mean',
        'temp_max': 'mean',
        'temp_min': 'mean',
        'daylight_duration': 'mean',
        'precipitation_sum': 'mean',
        'precipitation_hours': 'mean',
        'week_precipitation_interaction': 'mean',
        'daylight_precipitation_interaction': 'mean',
        'block_week_interaction': 'mean',
        'block_temp_max_interaction': 'mean',
        'temp_range': 'mean',
        'temp_range_precipitation_interaction': 'mean',
        'is_holiday': 'mean',
        'week_of_year': 'first'
    }

    # Group by 'census_block_numeric' and weekly intervals
    time_series = data.groupby(['census_block_numeric', pd.Grouper(freq='W')]).agg(agg_dict).reset_index()

    # Calculate the weekly crime count per census block
    time_series['Crime_Count_W'] = data.groupby(['census_block_numeric', pd.Grouper(freq='W')]).size().values

    # Apply log transformation to Crime_Count_W
    time_series['Crime_Count_W_log'] = np.log1p(time_series['Crime_Count_W'])

    # Proceed to create lag and rolling features on the log-transformed crime count
    target_column = 'Crime_Count_W_log'

    # Rename 'offense_date' back to 'date' if needed
    time_series.rename(columns={'offense_date': 'date'}, inplace=True)

    # Sort the time series
    time_series = time_series.sort_values(by=['census_block_numeric', 'date'])

    # Add lag features
    lag_intervals = [1, 4, 12, 24, 52]
    time_series = add_weekly_lag_features(time_series, target_column=target_column, lags=lag_intervals)

    # Add rolling features
    rolling_windows = [2, 3, 4, 8, 15]
    time_series = add_rolling_features(time_series, target_column=target_column, windows=rolling_windows)

    # Update interaction features to use the log-transformed features
    for window in rolling_windows:
        time_series[f'week_rolling{window}_interaction'] = (
                time_series['week_of_year'] * time_series[f'{target_column}_rolling{window}']
        )
        time_series[f'temp_rolling{window}_interaction'] = (
                time_series['temp_range'] * time_series[f'{target_column}_rolling{window}']
        )
        time_series[f'precip_rolling{window}_interaction'] = (
                time_series['precipitation_hours'] * time_series[f'{target_column}_rolling{window}']
        )

    for lag in lag_intervals:
        time_series[f'daylight_lag{lag}_interaction'] = (
                time_series['daylight_duration'] * time_series[f'{target_column}_lag{lag}w']
        )

    # Drop rows with NaN values resulting from lag and rolling operations
    time_series = time_series.dropna().reset_index(drop=True)

    # Save the time series data for later use
    time_series.to_csv('crime_time_series_nn.csv', index=False)

    return time_series


def categorize_crime_counts_training(train_data):
    """
    Categorize crime counts into 'low', 'medium', and 'high' buckets based on training data.

    Parameters:
    - train_data: DataFrame containing the 'Crime_Count_W' column.

    Returns:
    - Updated DataFrame with a new 'Crime_Level' column.
    - Thresholds used for categorization.
    """
    # Calculate percentiles on training data
    low_threshold = train_data['Crime_Count_W_log'].quantile(0.33)
    high_threshold = train_data['Crime_Count_W_log'].quantile(0.66)

    # Define function to assign categories
    def assign_category(count):
        if count <= low_threshold:
            return 0  # Low
        elif count <= high_threshold:
            return 1  # Medium
        else:
            return 2  # High

    # Apply the function to create a new column
    train_data['Crime_Level'] = train_data['Crime_Count_W_log'].apply(assign_category)

    # Save thresholds
    thresholds = (low_threshold, high_threshold)
    joblib.dump(thresholds, 'crime_level_thresholds.pkl')

    return train_data, thresholds


def assign_labels(data, thresholds):
    """
    Assign 'Crime_Level' labels to data using the thresholds from training data.
    """
    low_threshold, high_threshold = thresholds

    # Define function to assign categories
    def assign_category(count):
        if count <= low_threshold:
            return 0  # Low
        elif count <= high_threshold:
            return 1  # Medium
        else:
            return 2  # High

    # Apply the function to create a new column
    data['Crime_Level'] = data['Crime_Count_W_log'].apply(assign_category)

    return data


def shape_data(data, timesteps):
    """
    Creates sequences of temporal and spatial features for modeling.
    """
    # Define features for each model component
    spatial_features = ['census_block_numeric']  # Use encoded feature
    temporal_features = [
        "dow_cos", "dow_sin", "precipitation_hours",
        "Crime_Count_W_log_rolling15", "Crime_Count_W_log_rolling8", "Crime_Count_W_log_rolling4",
        "day_cos", "Crime_Count_W_log_rolling3", "Crime_Count_W_log_rolling2",
        "temp_rolling15_interaction", "temp_rolling4_interaction", "temp_rolling3_interaction",
        "temp_rolling2_interaction", "precip_rolling2_interaction", "Crime_Count_W_log_lag1w",
        "Crime_Count_W_log_lag4w", "day_sin", "daylight_lag1_interaction", "precip_rolling3_interaction",
        "daylight_lag4_interaction", "daylight_lag12_interaction", "daylight_lag24_interaction",
        "Crime_Count_W_log_lag52w", "precip_rolling4_interaction", "Crime_Count_W_log_lag12w",
        "precipitation_sum", "Crime_Count_W_log_lag24w", "daylight_lag52_interaction",
        "temp_range", "precip_rolling15_interaction"
    ]

    # Prepare data per census_block_numeric
    blocks = data['census_block_numeric'].unique()
    X_temporal_sequences_list = []
    X_spatial_sequences_list = []
    y_sequences_list = []

    for block in blocks:
        block_data = data[data['census_block_numeric'] == block].copy()
        # Sort by date
        block_data = block_data.sort_values('date')

        if len(block_data) < timesteps:
            continue  # Skip blocks with insufficient data

        X_temporal = block_data[temporal_features].values
        X_spatial = block_data[spatial_features].values
        y = block_data['Crime_Level'].values

        # Create sequences
        X_temporal_seq, X_spatial_seq, y_seq = [], [], []
        for i in range(len(X_temporal) - timesteps + 1):
            X_temporal_seq.append(X_temporal[i:(i + timesteps)])
            X_spatial_seq.append(X_spatial[i + timesteps - 1])  # Use spatial data at the prediction time
            y_seq.append(y[i + timesteps - 1])
        if len(X_temporal_seq) > 0:
            X_temporal_sequences_list.append(np.array(X_temporal_seq))
            X_spatial_sequences_list.append(np.array(X_spatial_seq))
            y_sequences_list.append(np.array(y_seq))

    # Concatenate sequences from all blocks
    if X_temporal_sequences_list:
        X_temporal_sequences = np.concatenate(X_temporal_sequences_list, axis=0)
        X_spatial_sequences = np.concatenate(X_spatial_sequences_list, axis=0)
        y_sequences = np.concatenate(y_sequences_list, axis=0)
    else:
        print("No data available after processing blocks.")
        return None, None, None

    # Verify shapes
    print("X_temporal_sequences shape:", X_temporal_sequences.shape)
    print("X_spatial_sequences shape:", X_spatial_sequences.shape)
    print("y_sequences shape:", y_sequences.shape)

    return X_temporal_sequences, X_spatial_sequences, y_sequences


def create_classification_model(temporal_input_shape, spatial_input_dim, spatial_embedding_dim=20, learning_rate=0.001):
    # Temporal input (Bidirectional LSTM)
    temporal_input = Input(shape=temporal_input_shape, name="Temporal_Input")
    x = Bidirectional(LSTM(128, return_sequences=True))(temporal_input)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(64))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Spatial input (Embedding)
    spatial_input = Input(shape=(1,), name="Spatial_Input")
    embedding = Embedding(
        input_dim=spatial_input_dim,
        output_dim=spatial_embedding_dim,
        name='CensusBlock_Embedding'
    )(spatial_input)
    embedding = Flatten()(embedding)
    y = Dense(64, activation='relu')(embedding)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)

    # Combine temporal and spatial paths
    combined = concatenate([x, y])
    combined = Dense(128, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)

    output = Dense(3, activation='softmax', name="Output")(combined)

    # Compile model
    model = Model(inputs=[temporal_input, spatial_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def modeling_with_scaling_classification(
        X_train_temporal, X_train_spatial, y_train,
        X_val_temporal, X_val_spatial, y_val,
        X_test_spatial,
        learning_rate
):
    """
    Builds and trains the classification model to predict crime levels with scaling and normalization.
    Incorporates embedding layers for categorical spatial features.
    """
    # Verify shapes
    print("X_train_temporal.shape:", X_train_temporal.shape)
    print("X_train_spatial.shape:", X_train_spatial.shape)
    print("y_train.shape:", y_train.shape)

    # Initialize scaler for temporal features
    scaler_temporal = RobustScaler()

    # Reshape temporal data for scaling: flatten the first two dimensions
    n_train, timesteps, n_features_temporal = X_train_temporal.shape
    X_train_temporal_flat = X_train_temporal.reshape(-1, n_features_temporal)
    scaler_temporal.fit(X_train_temporal_flat)

    # Apply scaling to training temporal data
    X_train_temporal_scaled = scaler_temporal.transform(X_train_temporal_flat).reshape(n_train, timesteps,
                                                                                       n_features_temporal)

    # Apply scaling to validation temporal data
    n_val = X_val_temporal.shape[0]
    X_val_temporal_flat = X_val_temporal.reshape(-1, n_features_temporal)
    X_val_temporal_scaled = scaler_temporal.transform(X_val_temporal_flat).reshape(n_val, timesteps,
                                                                                   n_features_temporal)

    # Spatial data remains encoded and not scaled
    X_train_spatial_scaled = X_train_spatial  # No scaling
    X_val_spatial_scaled = X_val_spatial  # No scaling

    # Use all spatial data from train, validation, and test sets to determine input_dim
    all_spatial_data = np.concatenate([X_train_spatial, X_val_spatial, X_test_spatial])
    spatial_input_dim = np.max(all_spatial_data) + 1  # Assuming label encoding starts at 0

    # Create and compile the model
    model = create_classification_model(
        temporal_input_shape=(timesteps, n_features_temporal),
        spatial_input_dim=spatial_input_dim,
        spatial_embedding_dim=20,  # Adjust embedding size as needed
        learning_rate=learning_rate
    )

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    # Train the model
    history = model.fit(
        [X_train_temporal_scaled, X_train_spatial_scaled],
        y_train,
        validation_data=(
            [X_val_temporal_scaled, X_val_spatial_scaled],
            y_val
        ),
        epochs=100,
        batch_size=64,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        class_weight=class_weights_dict,
        verbose=1,
    )

    # Plot training and validation loss
    plt.figure(figsize=(8, 4))
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model, scaler_temporal


def evaluate_model(model, scaler_temporal, X_test_temporal, X_test_spatial, y_test):
    """
    Evaluates the model on the test set and prints out metrics.
    """
    # Scale temporal features
    n_test, timesteps, n_features_temporal = X_test_temporal.shape
    X_test_temporal_flat = X_test_temporal.reshape(-1, n_features_temporal)
    X_test_temporal_scaled = scaler_temporal.transform(X_test_temporal_flat).reshape(n_test, timesteps,
                                                                                     n_features_temporal)

    # Spatial data remains encoded
    X_test_spatial_scaled = X_test_spatial  # No scaling

    # Predict on test data
    y_pred_prob = model.predict([X_test_temporal_scaled, X_test_spatial_scaled])
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")

    # Classification Report
    print("\nClassification Report:")
    class_names = ['Low', 'Medium', 'High']
    labels = [0, 1, 2]
    print("Unique classes in y_test:", np.unique(y_test))
    print("Unique classes in y_pred:", np.unique(y_pred))
    print(classification_report(y_test, y_pred, digits=4, target_names=class_names, labels=labels, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()


def predict_one_week_ahead(crime_time_series, timesteps, scaler_temporal_final, final_model, le_census_block):
    """
    Prepares data for predicting one week ahead of the latest date in the dataset.
    """
    # Define features
    temporal_features = [
        "dow_cos", "dow_sin", "precipitation_hours",
        "Crime_Count_W_log_rolling15", "Crime_Count_W_log_rolling8", "Crime_Count_W_log_rolling4",
        "day_cos", "Crime_Count_W_log_rolling3", "Crime_Count_W_log_rolling2",
        "temp_rolling15_interaction", "temp_rolling4_interaction", "temp_rolling3_interaction",
        "temp_rolling2_interaction", "precip_rolling2_interaction", "Crime_Count_W_log_lag1w",
        "Crime_Count_W_log_lag4w", "day_sin", "daylight_lag1_interaction", "precip_rolling3_interaction",
        "daylight_lag4_interaction", "daylight_lag12_interaction", "daylight_lag24_interaction",
        "Crime_Count_W_log_lag52w", "precip_rolling4_interaction", "Crime_Count_W_log_lag12w",
        "precipitation_sum", "Crime_Count_W_log_lag24w", "daylight_lag52_interaction",
        "temp_range", "precip_rolling15_interaction"
    ]
    spatial_features = ['census_block_numeric']

    # Find the maximum date in the dataset
    max_date = crime_time_series['date'].max()
    max_date = pd.to_datetime(max_date)

    # Prediction date is one week ahead of the maximum date
    prediction_date = max_date + pd.Timedelta(days=7)
    prediction_date_str = prediction_date.strftime('%Y-%m-%d')

    # Prepare data per census block
    blocks = crime_time_series['census_block_numeric'].unique()
    X_temporal_pred_list = []
    X_spatial_pred_list = []
    block_indices = []
    prediction_dates = []

    for block in blocks:
        block_data = crime_time_series[crime_time_series['census_block_numeric'] == block].copy()
        block_data = block_data.sort_values('date')

        # Filter data up to max_date
        block_data = block_data[block_data['date'] <= max_date]

        if len(block_data) < timesteps:
            continue  # Skip blocks with insufficient data

        # Get the last 'timesteps' records up to max_date
        recent_data = block_data.iloc[-timesteps:]

        if len(recent_data) != timesteps:
            continue  # Skip if not enough data

        X_temporal = recent_data[temporal_features].values

        # Use spatial data at the prediction time (assume it's the same as last known)
        X_spatial = recent_data[spatial_features].values[-1]  # Last known spatial data

        # Append prediction date (one week ahead of max_date)
        prediction_dates.append(prediction_date_str)

        X_temporal_pred_list.append(X_temporal)
        X_spatial_pred_list.append(X_spatial)
        block_indices.append(block)

    # Convert lists to arrays
    if not X_temporal_pred_list:
        print("No data available for prediction.")
        return

    X_temporal_pred = np.array(X_temporal_pred_list)
    X_spatial_pred = np.array(X_spatial_pred_list)

    # Scale temporal features
    n_pred_samples, pred_timesteps, n_features_temporal = X_temporal_pred.shape
    X_temporal_pred_flat = X_temporal_pred.reshape(-1, n_features_temporal)
    X_temporal_pred_scaled_flat = scaler_temporal_final.transform(X_temporal_pred_flat)
    X_temporal_pred_scaled = X_temporal_pred_scaled_flat.reshape(n_pred_samples, pred_timesteps, n_features_temporal)

    # Ensure X_spatial_pred_scaled has shape (n_samples, 1)
    X_spatial_pred_scaled = X_spatial_pred.reshape(-1, 1)

    # Verify input shapes
    print("X_temporal_pred_scaled shape:", X_temporal_pred_scaled.shape)
    print("X_spatial_pred_scaled shape:", X_spatial_pred_scaled.shape)

    # Make predictions
    y_pred_prob = final_model.predict([X_temporal_pred_scaled, X_spatial_pred_scaled])
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Map predictions to class names
    class_names = ['Low', 'Medium', 'High']
    y_pred_labels = [class_names[label] for label in y_pred]

    # Map census_block_numeric back to original census_block
    original_census_blocks = le_census_block.inverse_transform(X_spatial_pred_scaled.flatten().astype(int))

    # Create DataFrame with predictions
    prediction_results = pd.DataFrame({
        'census_block_numeric': X_spatial_pred_scaled.flatten(),
        'census_block': original_census_blocks,
        'Prediction_Date': prediction_dates,
        'Predicted_Crime_Level': y_pred_labels,
        'Prediction_Probability': y_pred_prob.max(axis=1)
    })

    # Display and save predictions
    print("\nPredictions for One Week Ahead:")
    print(prediction_results.head(10))  # Display first 10 predictions

    prediction_results.to_csv('one_week_ahead_predictions.csv', index=False)
    print("\nPredictions saved to 'one_week_ahead_predictions.csv'.")


def main():
    # Step 1: Load Data
    file_path = r"\Users\singh\Downloads\crime_with_weather.csv"  # Ensure correct file path
    crime_raw = get_data(file_path)

    # Step 2: Preprocess Data
    crime = preprocess_crime_data(crime_raw)
    print("Data Preprocessed.")

    # Step 3: Create Time Series Data
    crime_time_series = create_time_series(crime)
    print("Time Series Data Created.")

    # Step 4: Split Data into Train, Validation, and Test Sets
    # Sort the data by date
    crime_time_series = crime_time_series.sort_values('date')

    # Determine split indices
    total_samples = len(crime_time_series)
    train_size = int(0.6 * total_samples)
    val_size = int(0.2 * total_samples)

    # Split the data
    train_data = crime_time_series.iloc[:train_size]
    val_data = crime_time_series.iloc[train_size:train_size + val_size]
    test_data = crime_time_series.iloc[train_size + val_size:]

    # Step 5: Categorize Crime Counts into Crime Levels
    train_data, thresholds = categorize_crime_counts_training(train_data)
    # Load thresholds
    thresholds = joblib.load('crime_level_thresholds.pkl')
    le_census_block = joblib.load('le_census_block.pkl')

    # Assign labels to validation and test data
    val_data = assign_labels(val_data, thresholds)
    test_data = assign_labels(test_data, thresholds)

    # Step 6: Shape Data for Modeling
    timesteps = 1  # Adjust as needed
    X_train_temporal, X_train_spatial, y_train = shape_data(train_data, timesteps)
    X_val_temporal, X_val_spatial, y_val = shape_data(val_data, timesteps)
    X_test_temporal, X_test_spatial, y_test = shape_data(test_data, timesteps)

    # Check class distribution
    print("Training class distribution:", np.bincount(y_train))
    print("Validation class distribution:", np.bincount(y_val))
    print("Test class distribution:", np.bincount(y_test))

    if X_train_temporal is not None:
        # Step 7: Train the model with scaling
        model, scaler_temporal = modeling_with_scaling_classification(
            X_train_temporal, X_train_spatial, y_train,
            X_val_temporal, X_val_spatial, y_val,
            X_test_spatial,
            learning_rate=0.001
        )

        # Step 8: Evaluate the model on the test set
        evaluate_model(model, scaler_temporal,
                       X_test_temporal, X_test_spatial, y_test)

        # Step 9: Use the model to predict one week ahead
        predict_one_week_ahead(crime_time_series, timesteps, scaler_temporal, model, le_census_block)

        print("Final model used to predict one week into the future.")

        # Step 10: Save the final model and scaler
        model.save('final_classification_model.h5')
        joblib.dump(scaler_temporal, 'scaler_temporal_final.pkl')

        print("Final model and scaler saved successfully.")
    else:
        print("No data available for modeling.")


if __name__ == "__main__":
    main()
