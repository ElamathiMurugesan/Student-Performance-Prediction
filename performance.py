# ===============================
# Student Performance Project
# ===============================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# Step 1: Load Dataset
# -----------------------
df = pd.read_csv("student.csv")  # Replace with your dataset
print("Columns in dataset:", df.columns)
print(df.head())

# -----------------------
# Step 2: Create Target
# -----------------------
# Synthetic target since grades are missing
df['FinalScore'] = (
    df['StudyTimeWeekly']*10 +
    df['ParentalEducation']*10 -   # Combined Medu & Fedu assumption
    df['Absences']*2 +
    df['Age']
)
target_column = 'FinalScore'

# -----------------------
# Step 3: Preprocessing
# -----------------------
# Fill missing values
for col in df.select_dtypes(include=['float64','int64']).columns:
    df[col].fillna(df[col].mean(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Train / Validation / Test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -----------------------
# Step 4: EDA & Visualization
# -----------------------
# 1. Histogram
plt.figure(figsize=(8,5))
sns.histplot(df[target_column], bins=15, kde=True, color='skyblue')
plt.title("Distribution of Final Scores")
plt.show()

# 2. Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(y=df[target_column], color='lightgreen')
plt.title("Boxplot of Final Scores")
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 4. Scatterplot: StudyTimeWeekly vs FinalScore
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['StudyTimeWeekly'], y=df[target_column])
plt.title("StudyTimeWeekly vs FinalScore")
plt.show()

# 5. Countplot: ParentalEducation
plt.figure(figsize=(8,5))
sns.countplot(x=df['ParentalEducation'], palette='Set2')
plt.title("ParentalEducation Distribution")
plt.show()

# -----------------------
# Step 5: MLP Model
# -----------------------
input_dim = X_train.shape[1]
model = Sequential([
    Dense(64, input_dim=input_dim, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop]
)

# -----------------------
# Step 6: Evaluation & Visualization
# -----------------------
# 1. Loss vs Epoch
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss vs Epoch")
plt.legend()
plt.show()

# 2. MAE vs Epoch
plt.figure(figsize=(8,5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title("MAE vs Epoch")
plt.legend()
plt.show()

# 3. Actual vs Predicted
y_pred = model.predict(X_test)
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='purple')
plt.xlabel("Actual FinalScore")
plt.ylabel("Predicted FinalScore")
plt.title("Actual vs Predicted FinalScore")
plt.show()

# 4. Prediction Error Distribution
errors = y_test - y_pred.flatten()
plt.figure(figsize=(8,5))
sns.histplot(errors, bins=20, kde=True, color='red')
plt.title("Prediction Error Distribution")
plt.show()

# 5. Regression Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}, R²: {r2:.2f}")
