import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/Instagram_reach.csv")

# Preview structure
print(df.head())
print("\nColumns:", df.columns)
print("\nData Types:\n", df.dtypes)

# === Feature Engineering ===

# Fill missing values
df['Caption'] = df['Caption'].fillna("")
df['Hashtags'] = df['Hashtags'].fillna("")
df['Time since posted'] = df['Time since posted'].fillna("0 hours")

# Feature 1: Caption Length
df['CaptionLength'] = df['Caption'].apply(lambda x: len(str(x)))

# Feature 2: Hashtag Count
df['HashtagCount'] = df['Hashtags'].apply(lambda x: len(str(x).split(',')) if x else 0)

# Feature 3: Posted Hours Ago
df['PostedHoursAgo'] = df['Time since posted'].str.extract(r'(\d+)').astype(float).fillna(12)

# Remove outliers for 'Likes' and 'Followers'
df = df[df['Likes'] < 1000]
df = df[df['Followers'] < 10000]

# Final features and target
features = ['Followers', 'CaptionLength', 'HashtagCount', 'PostedHoursAgo']
target = 'Likes'

print("\nProcessed Features Preview:")
print(df[features + [target]].head())

# Prepare X and y
X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Improved Model Performance:")
print(f"R² Score: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save trained Linear Regression model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save fitted scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
