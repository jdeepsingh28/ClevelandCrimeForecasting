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
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, BatchNormalization, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


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

    # Rename 'offense_date' back to 'date' if needed
    time_series.rename(columns={'offense_date': 'date'}, inplace=True)

    # Sort the time series
    time_series = time_series.sort_values(by=['census_block_numeric', 'date'])

    # Add lag features
    lag_intervals = [1, 4, 12, 24, 52]
    time_series = add_weekly_lag_features(time_series, target_column='Crime_Count_W', lags=lag_intervals)

    # Add rolling features
    rolling_windows = [2, 3, 4, 8, 15]
    time_series = add_rolling_features(time_series, target_column='Crime_Count_W', windows=rolling_windows)

    # Create interaction features
    for window in rolling_windows:
        time_series[f'week_rolling{window}_interaction'] = (
            time_series['week_of_year'] * time_series[f'Crime_Count_W_rolling{window}']
        )
        time_series[f'temp_rolling{window}_interaction'] = (
            time_series['temp_range'] * time_series[f'Crime_Count_W_rolling{window}']
        )
        time_series[f'precip_rolling{window}_interaction'] = (
            time_series['precipitation_hours'] * time_series[f'Crime_Count_W_rolling{window}']
        )

    for lag in lag_intervals:
        time_series[f'daylight_lag{lag}_interaction'] = (
            time_series['daylight_duration'] * time_series[f'Crime_Count_W_lag{lag}w']
        )

    # Drop rows with NaN values resulting from lag and rolling operations
    time_series = time_series.dropna().reset_index(drop=True)

    # Categorize crime counts into buckets
    time_series = categorize_crime_counts(time_series)

    # Verify the resulting DataFrame
    print("Time series data after weekly aggregation and feature engineering:")
    print(time_series.head())

    return time_series

def categorize_crime_counts(time_series):
    """
    Categorize crime counts into 'low', 'medium', and 'high' buckets.

    Parameters:
    - time_series: DataFrame containing the 'Crime_Count_W' column.

    Returns:
    - Updated DataFrame with a new 'Crime_Level' column.
    """
    # Calculate percentiles
    low_threshold = time_series['Crime_Count_W'].quantile(0.33)
    high_threshold = time_series['Crime_Count_W'].quantile(0.66)

    # Define function to assign categories
    def assign_category(count):
        if count <= low_threshold:
            return 0  # Low
        elif count <= high_threshold:
            return 1  # Medium
        else:
            return 2  # High

    # Apply the function to create a new column
    time_series['Crime_Level'] = time_series['Crime_Count_W'].apply(assign_category)
    return time_series

def shape_data(data, timesteps):
    """
    Creates sequences of temporal and spatial features for modeling.
    """
    # Define features for each model component
    spatial_features = ['census_block_numeric']  # Use encoded feature
    temporal_features = [
        "dow_cos", "dow_sin", "precipitation_hours",
        "Crime_Count_W_rolling15", "Crime_Count_W_rolling8", "Crime_Count_W_rolling4",
        "day_cos", "Crime_Count_W_rolling3", "Crime_Count_W_rolling2",
        "temp_rolling15_interaction", "temp_rolling4_interaction", "temp_rolling3_interaction",
        "temp_rolling2_interaction", "precip_rolling2_interaction", "Crime_Count_W_lag1w",
        "Crime_Count_W_lag4w", "day_sin", "daylight_lag1_interaction", "precip_rolling3_interaction",
        "daylight_lag4_interaction", "daylight_lag12_interaction", "daylight_lag24_interaction",
        "Crime_Count_W_lag52w", "precip_rolling4_interaction", "Crime_Count_W_lag12w",
        "precipitation_sum", "Crime_Count_W_lag24w", "daylight_lag52_interaction",
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

def create_classification_model(temporal_input_shape, spatial_input_dim, spatial_embedding_dim=16, learning_rate=0.001):
    # Temporal input (LSTM)
    temporal_input = Input(shape=temporal_input_shape, name="Temporal_Input")
    x = LSTM(128, return_sequences=True)(temporal_input)
    x = BatchNormalization()(x)
    x = LSTM(64)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Spatial input (Embedding)
    spatial_input = Input(shape=(1,), name="Spatial_Input")  # Expecting shape (batch_size, 1)
    embedding = Embedding(input_dim=spatial_input_dim, output_dim=spatial_embedding_dim, input_length=1)(spatial_input)
    embedding = Flatten()(embedding)  # Shape: (batch_size, embedding_dim)
    y = Dense(64, activation='relu')(embedding)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)

    # Combine temporal and spatial paths
    combined = concatenate([x, y])
    combined = Dense(64, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.3)(combined)
    output = Dense(3, activation='softmax', name="Output")(combined)

    # Compile model
    model = Model(inputs=[temporal_input, spatial_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def modeling_with_scaling_classification(X_temporal_sequences, X_spatial_sequences, y_sequences, learning_rate):
    """
    Builds and trains the classification model to predict crime levels with scaling and normalization.
    Incorporates embedding layers for categorical spatial features.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, BatchNormalization, Embedding, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import RobustScaler

    # Verify shapes
    print("X_temporal_sequences.shape:", X_temporal_sequences.shape)
    print("X_spatial_sequences.shape:", X_spatial_sequences.shape)
    print("y_sequences.shape:", y_sequences.shape)

    # Get the total number of unique census blocks for embedding input_dim
    spatial_input_dim = np.max(X_spatial_sequences) + 1  # Assuming label encoding starts at 0

    # TimeSeriesSplit cross-validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Initialize lists to store all results
    all_y_val = []
    all_y_pred = []
    all_y_pred_prob = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    train_losses = []
    val_losses = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X_temporal_sequences)):
        print(f"\nProcessing fold {fold + 1}/{n_splits}")

        # Split the sequences
        X_temporal_fold_train = X_temporal_sequences[train_index]
        X_temporal_fold_val = X_temporal_sequences[val_index]
        X_spatial_fold_train = X_spatial_sequences[train_index]
        X_spatial_fold_val = X_spatial_sequences[val_index]
        y_fold_train = y_sequences[train_index]
        y_fold_val = y_sequences[val_index]

        # Initialize scaler for temporal features
        scaler_temporal = RobustScaler()

        # Reshape temporal data for scaling: flatten the first two dimensions
        n_train, timesteps, n_features_temporal = X_temporal_fold_train.shape
        X_temporal_fold_train_flat = X_temporal_fold_train.reshape(-1, n_features_temporal)
        scaler_temporal.fit(X_temporal_fold_train_flat)

        # Apply scaling to training temporal data
        X_temporal_fold_train_scaled = scaler_temporal.transform(X_temporal_fold_train_flat).reshape(n_train, timesteps, n_features_temporal)

        # Apply scaling to validation temporal data
        n_val = X_temporal_fold_val.shape[0]
        X_temporal_fold_val_flat = X_temporal_fold_val.reshape(-1, n_features_temporal)
        X_temporal_fold_val_scaled = scaler_temporal.transform(X_temporal_fold_val_flat).reshape(n_val, timesteps, n_features_temporal)

        # Spatial data remains encoded and not scaled
        X_spatial_fold_train_scaled = X_spatial_fold_train  # No scaling
        X_spatial_fold_val_scaled = X_spatial_fold_val      # No scaling

        # Create and compile the model
        model = create_classification_model(
            temporal_input_shape=(timesteps, n_features_temporal),
            spatial_input_dim=spatial_input_dim,
            spatial_embedding_dim=25,  # Adjust embedding size as needed
            learning_rate=learning_rate
        )

        # Train the model
        history = model.fit(
            [X_temporal_fold_train_scaled, X_spatial_fold_train_scaled],
            y_fold_train,
            validation_data=(
                [X_temporal_fold_val_scaled, X_spatial_fold_val_scaled],
                y_fold_val
            ),
            epochs=100,
            batch_size=64,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=1,
        )


        # Collect training and validation loss history
        train_losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])

        # Predict on validation data
        y_pred_prob = model.predict([X_temporal_fold_val_scaled, X_spatial_fold_val_scaled])
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Append validation labels and predictions
        all_y_val.extend(y_fold_val)
        all_y_pred.extend(y_pred)
        all_y_pred_prob.append(y_pred_prob)

        # Calculate metrics for the current fold
        acc = accuracy_score(y_fold_val, y_pred)
        precision = precision_score(y_fold_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_fold_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_fold_val, y_pred, average='weighted', zero_division=0)
        accuracy_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # Concatenate all predicted probabilities
    all_y_pred_prob = np.concatenate(all_y_pred_prob, axis=0)

    return accuracy_scores, precision_scores, recall_scores, f1_scores, all_y_val, all_y_pred, all_y_pred_prob, train_losses, val_losses


def modeling_metrics_classification(accuracy_scores, precision_scores, recall_scores, f1_scores, all_y_val, all_y_pred, all_y_pred_prob, train_losses, val_losses):
    """
    Displays classification metrics and plots for model evaluation.
    """
    # Combine metrics for reporting
    print(f"\nAverage Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Average Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Average Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"Average F1-score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_y_val, all_y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_y_val, all_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot training and validation loss for each fold
    for i in range(len(train_losses)):
        plt.figure(figsize=(8, 4))
        epochs = range(1, len(train_losses[i]) + 1)
        plt.plot(epochs, train_losses[i], 'b-', label='Training Loss')
        plt.plot(epochs, val_losses[i], 'r-', label='Validation Loss')
        plt.title(f'Training and Validation Loss - Fold {i+1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # Compute ROC curve and ROC area for each class
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    # Binarize the output
    classes = [0, 1, 2]
    y_test = label_binarize(all_y_val, classes=classes)
    n_classes = y_test.shape[1]
    y_score = all_y_pred_prob  # Shape: (n_samples, n_classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Print AUC Scores
    print("\nROC AUC Scores:")
    for i in range(n_classes):
        print(f"Class {i} AUC: {roc_auc[i]:.4f}")
    print(f"Macro-average AUC: {roc_auc['macro']:.4f}")
    print(f"Micro-average AUC: {roc_auc['micro']:.4f}")

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    class_names = ['Low', 'Medium', 'High']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def predict_one_week_ahead(crime_time_series, timesteps, scaler_temporal_final, final_model):
    """
    Prepares data for one-week-ahead prediction and makes predictions using the trained model.
    """
    # Define features
    temporal_features = [
        "dow_cos", "dow_sin", "precipitation_hours",
        "Crime_Count_W_rolling15", "Crime_Count_W_rolling8", "Crime_Count_W_rolling4",
        "day_cos", "Crime_Count_W_rolling3", "Crime_Count_W_rolling2",
        "temp_rolling15_interaction", "temp_rolling4_interaction", "temp_rolling3_interaction",
        "temp_rolling2_interaction", "precip_rolling2_interaction", "Crime_Count_W_lag1w",
        "Crime_Count_W_lag4w", "day_sin", "daylight_lag1_interaction", "precip_rolling3_interaction",
        "daylight_lag4_interaction", "daylight_lag12_interaction", "daylight_lag24_interaction",
        "Crime_Count_W_lag52w", "precip_rolling4_interaction", "Crime_Count_W_lag12w",
        "precipitation_sum", "Crime_Count_W_lag24w", "daylight_lag52_interaction",
        "temp_range", "precip_rolling15_interaction"
    ]
    spatial_features = ['census_block_numeric']

    # Prepare data per census block
    blocks = crime_time_series['census_block_numeric'].unique()
    X_temporal_pred_list = []
    X_spatial_pred_list = []
    block_indices = []

    for block in blocks:
        block_data = crime_time_series[crime_time_series['census_block_numeric'] == block].copy()
        block_data = block_data.sort_values('date')

        if len(block_data) < timesteps:
            continue  # Skip blocks with insufficient data

        # Get the most recent 'timesteps' records
        recent_data = block_data.iloc[-timesteps:]

        if len(recent_data) != timesteps:
            continue  # Skip if not enough data

        X_temporal = recent_data[temporal_features].values
        X_spatial = recent_data[spatial_features].values[-1]  # Use spatial data at the prediction time

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

    # Spatial data remains encoded
    X_spatial_pred_scaled = X_spatial_pred  # No scaling

    # Make predictions
    y_pred_prob = final_model.predict([X_temporal_pred_scaled, X_spatial_pred_scaled])
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Map predictions to class names
    class_names = ['Low', 'Medium', 'High']
    y_pred_labels = [class_names[label] for label in y_pred]

    # Create DataFrame with predictions
    prediction_results = pd.DataFrame({
        'census_block_numeric': block_indices,
        'Predicted_Crime_Level': y_pred_labels,
        'Prediction_Probability': y_pred_prob.max(axis=1)
    })

    # Display and save predictions
    print("\nOne-Week-Ahead Predictions:")
    print(prediction_results)  # Display first 10 predictions

    prediction_results.to_csv('one_week_ahead_predictions-NN.csv', index=False)
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

    # Step 4: Shape Data for Modeling
    timesteps = 1  # Adjust as needed
    X_temporal_sequences, X_spatial_sequences, y_sequences = shape_data(crime_time_series, timesteps)

    if X_temporal_sequences is not None:
        # Step 5: Train the model and get predictions with scaling
        accuracy_scores, precision_scores, recall_scores, f1_scores, all_y_val, all_y_pred, all_y_pred_prob, train_losses, val_losses = modeling_with_scaling_classification(
            X_temporal_sequences, X_spatial_sequences, y_sequences, learning_rate=0.001
        )

        # Step 6: Display metrics and plots
        modeling_metrics_classification(accuracy_scores, precision_scores, recall_scores, f1_scores, all_y_val,
                                        all_y_pred, all_y_pred_prob, train_losses, val_losses)

        # Step 7: Train a final model on the entire dataset
        # Fit scaler on the entire temporal data
        n_samples, timesteps, n_features_temporal = X_temporal_sequences.shape
        X_temporal_flat = X_temporal_sequences.reshape(-1, n_features_temporal)
        scaler_temporal_final = RobustScaler()
        scaler_temporal_final.fit(X_temporal_flat)
        X_temporal_scaled_final = scaler_temporal_final.transform(X_temporal_flat).reshape(n_samples, timesteps,
                                                                                           n_features_temporal)
        # Spatial data remains encoded and not scaled
        X_spatial_scaled_final = X_spatial_sequences  # No scaling

        # Create and compile the final model
        final_model = create_classification_model(
            temporal_input_shape=(timesteps, n_features_temporal),
            spatial_input_dim=np.max(X_spatial_sequences) + 1,  # Number of unique census blocks
            spatial_embedding_dim=25,  # Adjust embedding size as needed
            learning_rate=0.001
        )

        # Train the final model
        history_final = final_model.fit(
            [X_temporal_scaled_final, X_spatial_scaled_final],
            y_sequences,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=1,
        )

        # Save the final model and scaler
        final_model.save('final_cnn_rnn_classification_model.h5')
        joblib.dump(scaler_temporal_final, 'scaler_temporal_final.pkl')

        print("Final model and scaler saved successfully.")

        predict_one_week_ahead(crime_time_series, timesteps, scaler_temporal_final, final_model)

        print("Final model used to predict one week into the future.")
    else:
        print("No data available for modeling.")


if __name__ == "__main__":
    main()
