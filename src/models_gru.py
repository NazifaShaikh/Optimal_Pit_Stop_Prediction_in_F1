# Core numerical + plotting
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn utilities
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1) Load preprocessed sequences and ensure numeric dtypes

data = np.load("data/f1_sequences.npz", allow_pickle=True)
X, y = data['X'], data['y']

# Cast to float32 to avoid object dtype errors
if X.dtype == object:
    X = np.array([seq.astype(np.float32) for seq in X], dtype=np.float32)
else:
    X = X.astype(np.float32)

y = y.astype(np.float32)

print("X shape:", X.shape, "y shape:", y.shape)
print("X dtype:", X.dtype, "y dtype:", y.dtype)

# 2) Train/validation/test split
# Random split for now; race-level split can be added later
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# 3) Build improved GRU model

model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),   # (timesteps, features)

    # Bidirectional GRU captures both past and future context
    Bidirectional(GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),

    #   Second GRU layer for deeper sequence learning
    GRU(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),

    # Dense layers with dropout for regularization
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(1)  # regression output: next pit lap
])

# Optimizer: RMSprop often works well for RNNs
optimizer = RMSprop(learning_rate=0.0005)

model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mae'])

# 4) Train model with callbacks

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# 5) Predictions on test set

y_pred = model.predict(X_test).reshape(-1)

# 6) Evaluation metrics (MAE, MSE, RMSE, MAPE)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # manual RMSE for compatibility

print(f"MAE (test):  {mae:.3f} laps")
print(f"MSE (test):  {mse:.3f}")
print(f"RMSE (test): {rmse:.3f} laps")

# 7) Visualizations

# --- Training history ---
plt.figure(figsize=(12,5))

# Loss curve
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Val MSE')
plt.xlabel('Epoch \n (a)'); plt.ylabel('MSE')
plt.title('Training vs Validation Loss')
plt.legend()

# MAE curve
plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Epoch \n (b)'); plt.ylabel('MAE (laps)')
plt.title('Training vs Validation MAE')
plt.legend()

plt.tight_layout()
plt.show()

# --- Prediction scatter plot ---
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.35, edgecolor='none')
min_v, max_v = float(np.min(y_test)), float(np.max(y_test))
plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect fit')
plt.xlabel("True Next Pit Lap"); plt.ylabel("Predicted Next Pit Lap")
plt.title("GRU Predictions vs True Values")
plt.legend()
plt.show()

# --- Residuals distribution ---
residuals = y_pred - y_test
plt.figure(figsize=(10,4))
plt.hist(residuals, bins=40, alpha=0.8, color='steelblue')
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel("Residual (prediction - truth) [laps]")
plt.ylabel("Count")
plt.title("Residual distribution on test set")
plt.show()