# XGBoost Hotel Booking Cancellation Prediction

This project provides a wrapper around an XGBoost model to predict hotel booking cancellations. It includes data preprocessing, one-hot encoding for categorical features, and prediction endpoints for both single and batch (file) inputs.

## Features

- Predicts hotel booking cancellations using an XGBoost model.
- Handles both single prediction and batch predictions from CSV files.
- Preprocesses input data, including one-hot encoding of categorical features.
- Returns prediction probabilities and, for batch predictions, client information.

## Project Structure

- `src/wrapper.py`: Main model wrapper for loading, preprocessing, and predicting.
- `src/dtos.py`: Data Transfer Objects (DTOs) for input and output schemas.
- `type_adapter.py`: (Contextual) Utilities for JSON schema and serialization.

## Requirements

- Python 3.10+
- pandas
- xgboost
- scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Model Initialization

```python
from src.wrapper import XGBoostModelWrapper

model = XGBoostModelWrapper('path/to/model.pkl', 'path/to/encoder.pkl')
```

### Single Prediction

```python
features = {
    "lead_time": 34,
    "hotel": "Resort Hotel",
    ...
}
result = model.predict_one(features)
print(result)
```

### Batch Prediction from File

```python
with open('input.csv', 'rb') as f:
    class FileObj:
        file = f
        filename = 'input.csv'
    result = model.predict_from_file(FileObj())
    print(result)
```

## Input Format

- For single predictions, provide a dictionary matching the fields in `FeaturesDto`.
- For batch predictions, provide a CSV file with columns matching the features.

## Output

- Single prediction: `OnePredictionOutputDto` with prediction and probabilities.
- Batch prediction: `FilePredictionOutputDto` with predictions, client info, and file metadata.

## License

MIT License



## Data example for one prediction

```{
  "lead_time": 342,
  "arrival_date_year": 2015,
  "arrival_date_week_number": 27,
  "arrival_date_day_of_month": 1,
  "stays_in_weekend_nights": 0,
  "stays_in_week_nights": 0,
  "adults": 2,
  "children": 0,
  "babies": 0,
  "is_repeated_guest": 0,
  "previous_cancellations": 0,
  "previous_bookings_not_canceled": 0,
  "booking_changes": 3,
  "days_in_waiting_list": 0,
  "adr": 0,
  "required_car_parking_spaces": 0,
  "total_of_special_requests": 0,
  "hotel": "Resort Hotel",
  "arrival_date_month": "July",
  "meal": "BB",
  "country": "PRT",
  "market_segment": "Direct",
  "distribution_channel": "Direct",
  "reserved_room_type": "C",
  "assigned_room_type": "C",
  "deposit_type": "No Deposit",
  "customer_type": "Transient"
}
```

## Docker commands:

For creating the container for the project use the following commands.

```
    docker build -t project-backend .
    docker run --name backend-project -p 8000:8000 -d project-backend
```
