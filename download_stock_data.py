import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Configuración para usar GPU si está disponible
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU encontrada y memoria de GPU configurada para crecimiento dinámico.")
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

def create_sequences(data, seq_length_val):
    """
    Crea secuencias de datos y sus etiquetas correspondientes para el modelo LSTM.
    """
    X, y = [], []
    # Asegurarse de que hay suficientes datos para al menos una secuencia y una etiqueta
    if len(data) <= seq_length_val:
        return np.array([]), np.array([])
        
    for i in range(len(data) - seq_length_val):
        X.append(data[i:i + seq_length_val])
        y.append(data[i + seq_length_val])
    return np.array(X), np.array(y)

def train_and_predict_lstm(ticker_name, period="2y", seq_length=30, epochs=2, batch_size=256):
    """
    Descarga datos históricos de un ticker, entrena un modelo LSTM
    y genera predicciones históricas.
    Devuelve también el modelo y el scaler para futuras predicciones.
    """
    print(f"Descargando datos para {ticker_name} (período: {period})...")
    try:
        data = yf.download(ticker_name, period=period)
        if data.empty:
            print(f"No hay datos disponibles para {ticker_name} en el período {period}.")
            return None, None, None, None, None, None, None
    except Exception as e:
        print(f"Error al descargar datos para {ticker_name}: {e}")
        return None, None, None, None, None, None, None

    if 'Close' not in data.columns:
        print(f"La columna 'Close' no se encontró en los datos para {ticker_name}.")
        return None, None, None, None, None, None, None
        
    # Preprocesar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    # Verificar que haya suficientes datos para el seq_length proporcionado
    if len(scaled_data) < seq_length + 1:
        print(f"Datos insuficientes para crear al menos una secuencia con seq_length={seq_length} para {ticker_name}.")
        return None, None, None, None, None, None, None

    # Crear conjuntos de entrenamiento y prueba
    train_size = int(len(scaled_data) * 0.9)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Asegúrate de tener suficientes datos para crear secuencias en ambos conjuntos
    # Se ajusta la lógica para verificar que haya al menos seq_length + 1 elementos para formar una secuencia y su etiqueta
    if len(train_data) <= seq_length or len(test_data) <= seq_length:
        print(f"Datos insuficientes para crear secuencias LSTM con seq_length={seq_length} para {ticker_name}. "
              f"Tamaño de entrenamiento: {len(train_data)}, Tamaño de prueba: {len(test_data)}")
        return None, None, None, None, None, None, None

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Verificar si se pudieron crear secuencias
    if X_train.size == 0 or X_test.size == 0:
        print(f"No se pudieron crear secuencias de entrenamiento/prueba con seq_length={seq_length} para {ticker_name}. "
              f"Verifique la longitud de los datos o reduzca seq_length.")
        return None, None, None, None, None, None, None

    # Redimensionar para la entrada de LSTM [muestras, timesteps, características]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = tf.keras.Sequential([
        # Primera capa LSTM: Más unidades, con regularización L2 para combatir sobreajuste
        tf.keras.layers.LSTM(128, return_sequences=True,
                             input_shape=(X_train.shape[1], 1),
                             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3), # Aumento del Dropout para mayor regularización

        # Segunda capa LSTM: Más unidades, también con regularización L2
        tf.keras.layers.LSTM(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3), # Aumento del Dropout

        # Capas Dense (Densely Connected): Aumentar la complejidad de la red
        tf.keras.layers.Dense(64, activation='relu'), # Capa oculta adicional con activación ReLU
        tf.keras.layers.Dropout(0.2), # Dropout antes de la capa final

        tf.keras.layers.Dense(1) # Capa de salida
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    print(f"Entrenando modelo LSTM para {ticker_name} (seq_length={seq_length})...")
    history = model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=epochs, workers=6, batch_size=batch_size, verbose=0, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ])
    print(f"Entrenamiento completado para {ticker_name}. Pérdida final (validación): {history.history['val_loss'][-1]:.4f}")

    # Hacer predicciones históricas
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Invertir la escala de las predicciones
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
    # Invertir la escala de los datos originales
    original_data_scaled_inv = scaler.inverse_transform(scaled_data)

    # Devolvemos el modelo y el scaler
    return original_data_scaled_inv, train_predict, test_predict, seq_length, data, model, scaler

def predict_future_prices(model, scaler, last_data_points, seq_length, num_future_steps):
    """
    Realiza predicciones de precios para los próximos N pasos en el futuro.

    Args:
        model (tf.keras.Model): El modelo LSTM entrenado.
        scaler (sklearn.preprocessing.MinMaxScaler): El scaler usado para normalizar los datos.
        last_data_points (np.array): Los últimos `seq_length` puntos de datos históricos (sin escalar).
        seq_length (int): La longitud de la secuencia que el modelo LSTM espera.
        num_future_steps (int): El número de pasos a predecir en el futuro.

    Returns:
        np.array: Un array con los precios predichos para los próximos `num_future_steps`.
    """
    if model is None or scaler is None or last_data_points is None or len(last_data_points) < seq_length:
        print("Datos insuficientes o modelo/scaler no proporcionado para predicción futura.")
        return np.array([])

    # Normalizar los últimos puntos de datos conocidos
    current_sequence = scaler.transform(last_data_points.reshape(-1, 1))
    current_sequence = current_sequence.reshape(1, seq_length, 1) # Redimensionar para la entrada del modelo

    future_predictions = []

    for _ in range(num_future_steps):
        # Predecir el siguiente paso
        next_prediction = model.predict(current_sequence, verbose=0)[0, 0] # Obtener el valor predicho

        future_predictions.append(next_prediction)

        # Actualizar la secuencia para la próxima predicción
        # Eliminar el primer elemento y añadir la nueva predicción
        next_prediction_reshaped = np.array([[[next_prediction]]])
        new_sequence = np.append(current_sequence[:, 1:, :], next_prediction_reshaped, axis=1)

        current_sequence = new_sequence
    
    # Invertir la escala de todas las predicciones futuras
    future_predictions_scaled_inv = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return future_predictions_scaled_inv.flatten() # Devolver como un array 1D
