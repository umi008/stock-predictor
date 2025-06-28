import customtkinter
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime 
import pandas as pd 

# Importamos las funciones desde nuestro nuevo archivo
from download_stock_data import train_and_predict_lstm, predict_future_prices

class App(customtkinter.CTk):
    """
    Clase principal de la aplicación CustomTkinter, ahora con integración de yfinance
    y un modelo LSTM modular para mostrar predicciones de precios, incluyendo el futuro.
    """
    def __init__(self):
        super().__init__()

        self.title("App de Tickers con Predicción LSTM")
        self.geometry("1000x700")
        self.protocol("WM_DELETE_WINDOW", self.destroy)  # Asegura que la ventana se cierre correctamente

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        customtkinter.set_appearance_mode("Dark")
        customtkinter.set_default_color_theme("blue")

        self.tickers = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK-B", "JPM", "JNJ",
            "V", "PG", "UNH", "HD", "MA", "XOM", "PFE", "KO", "PEP", "ABBV",
            "LLY", "MRK", "BAC", "COST", "DIS", "TMO", "AVGO", "CSCO", "ADBE", "CMCSA",
            "NFLX", "ABT", "INTC", "CRM", "WMT", "CVX", "NKE", "TXN", "QCOM", "AMD",
            "AMGN", "MDT", "HON", "UNP", "NEE", "PM", "LIN", "ORCL", "UPS", "MS"
        ]

        # =====================================================================
        # Sidebar Frame (sin cambios significativos en la estructura)
        # =====================================================================
        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=1, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1) 

        self.sidebar_label = customtkinter.CTkLabel(self.sidebar_frame, text="Tickers", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.sidebar_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.search_entry = customtkinter.CTkEntry(self.sidebar_frame, placeholder_text="Buscar ticker...")
        self.search_entry.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.search_entry.bind("<KeyRelease>", self._filter_sidebar_items)

        # OptionMenu para el rango de tiempo
        self.view_span_options = ["5 Años", "1 Año", "6 Meses", "3 Meses", "1 Mes"]
        self.selected_view_span = customtkinter.StringVar(value=self.view_span_options[0]) 
        self.view_span_menu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                          values=self.view_span_options,
                                                          command=self._on_view_span_change,
                                                          variable=self.selected_view_span)
        self.view_span_menu.grid(row=3, column=0, padx=20, pady=10, sticky="ew")


        self.scrollable_frame = customtkinter.CTkScrollableFrame(self.sidebar_frame, label_text="Lista de Tickers")
        self.scrollable_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        self.sidebar_buttons = {}
        self._create_sidebar_items()

        # =====================================================================
        # Main Content Frame (para la gráfica)
        # =====================================================================
        self.main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.current_selected_ticker = None # Para recordar el último ticker seleccionado

        if self.tickers:
            self._update_graph(self.tickers[0])
            self.current_selected_ticker = self.tickers[0]
        else:
            self._update_graph(None)

    def _create_sidebar_items(self):
        for i, ticker in enumerate(self.tickers):
            button = customtkinter.CTkButton(self.scrollable_frame,
                                             text=ticker,
                                             command=lambda name=ticker: self._on_sidebar_item_click(name))
            button.grid(row=i, column=0, padx=5, pady=5, sticky="ew")
            self.sidebar_buttons[ticker] = button

    def _filter_sidebar_items(self, event=None):
        search_term = self.search_entry.get().lower()
        for ticker, button in self.sidebar_buttons.items():
            if search_term in ticker.lower():
                button.grid()
            else:
                button.grid_remove()

    def _on_sidebar_item_click(self, ticker_name):
        print(f"Se hizo clic en: {ticker_name}")
        self.current_selected_ticker = ticker_name # Actualizar el ticker seleccionado
        self._update_graph(ticker_name)

    def _on_view_span_change(self, new_view_span):
        print(f"Rango de tiempo seleccionado: {new_view_span}")
        # Si hay un ticker seleccionado, actualizar la gráfica con el nuevo rango
        if self.current_selected_ticker:
            self._update_graph(self.current_selected_ticker)


    def _update_graph(self, ticker_name):
        """
        Actualiza la gráfica de matplotlib con datos históricos, predicciones históricas
        y predicciones futuras del modelo LSTM, aplicando el filtro de tiempo.
        """
        self.ax.clear() # Limpia la gráfica existente

        if ticker_name:
            selected_period_display = self.selected_view_span.get() # Período seleccionado para la visualización
            
            # Siempre descargar datos por 5 años para el entrenamiento
            yfinance_period_train = "5y" 
            
            # Ajustar seq_length basándose en el período de VISUALIZACIÓN para una mejor adaptación del modelo
            # aunque se entrene con 5 años de datos.
            calculated_seq_length = 30 # Seq_length por defecto

            if selected_period_display == "5 Años":
                calculated_seq_length = 30
            elif selected_period_display == "1 Año":
                calculated_seq_length = 20
            elif selected_period_display == "6 Meses":
                calculated_seq_length = 15
            elif selected_period_display == "3 Meses":
                calculated_seq_length = 10
            elif selected_period_display == "1 Mes":
                calculated_seq_length = 5
            
            # Asegurarse de que seq_length sea al menos 1
            calculated_seq_length = max(1, calculated_seq_length)


            # Llama a la función modular train_and_predict_lstm con el período fijo de 5 años
            # y el seq_length dinámico.
            original_data_scaled_inv, train_predict, test_predict, seq_length, data, model, scaler = \
                train_and_predict_lstm(ticker_name, period=yfinance_period_train, seq_length=calculated_seq_length, epochs=30) 

            if data is not None and not data.empty and original_data_scaled_inv is not None:
                # --- Calcular el rango de fechas para la visualización ---
                end_date = data.index.max()
                start_date = end_date # Valor por defecto, se ajustará

                if selected_period_display == "5 Años":
                    start_date = end_date - pd.DateOffset(years=5)
                elif selected_period_display == "1 Año":
                    start_date = end_date - pd.DateOffset(years=1)
                elif selected_period_display == "6 Meses":
                    start_date = end_date - pd.DateOffset(months=6)
                elif selected_period_display == "3 Meses":
                    start_date = end_date - pd.DateOffset(months=3)
                elif selected_period_display == "1 Mes":
                    start_date = end_date - pd.DateOffset(months=1)
                
                # Asegurarse de que start_date no sea anterior a la primera fecha disponible de los datos descargados
                start_date = max(start_date, data.index.min())
                
                # Filtrar los datos REALES para la visualización (el "zoom")
                display_data = data[data.index >= start_date]

                # Graficar los datos reales filtrados para la visualización
                self.ax.plot(display_data.index, display_data["Close"].values, label="Precio Real", color='blue')

                # --- Predicciones Históricas (Entrenamiento y Prueba) ---
                if train_predict is not None and test_predict is not None and seq_length is not None:
                    # Crear series de Pandas para las predicciones con el índice de tiempo correcto
                    # para los datos de ENTRENAMIENTO (basado en el índice original de 'data')
                    train_dates = data.index[seq_length:len(train_predict) + seq_length]
                    train_series = pd.Series(train_predict.flatten(), index=train_dates)
                    
                    # Para los datos de PRUEBA
                    test_start_index = len(train_predict) + seq_length
                    if test_start_index < len(data.index): # Asegurarse de que el índice no esté fuera de rango
                        test_dates = data.index[test_start_index:test_start_index + len(test_predict)]
                        test_series = pd.Series(test_predict.flatten(), index=test_dates)
                    else:
                        test_series = pd.Series([]) # No hay datos de prueba válidos para graficar
                        print("No hay suficientes datos para graficar la predicción de prueba.")

                    # Filtrar las predicciones históricas para el rango de visualización
                    filtered_train_series = train_series[train_series.index >= start_date]
                    filtered_test_series = test_series[test_series.index >= start_date]

                    self.ax.plot(filtered_train_series.index, filtered_train_series.values, label="Predicción Entrenamiento", color='green', linestyle='--')
                    if not filtered_test_series.empty: 
                        self.ax.plot(filtered_test_series.index, filtered_test_series.values, label="Predicción Prueba", color='red', linestyle='--')
                else:
                    self.ax.text(0.5, 0.5, "No se pudo generar la predicción LSTM histórica (datos insuficientes o error)",
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=self.ax.transAxes, fontsize=10, color='gray')

                # --- Predicciones Futuras ---
                if model is not None and scaler is not None and not data["Close"].empty and seq_length is not None:
                    # Obtener los últimos `seq_length` puntos de datos reales para la predicción futura
                    if len(data["Close"]) >= seq_length:
                        last_real_data = data["Close"].values[-seq_length:]
                        
                        num_future_days = 30 # Número de días a predecir en el futuro
                        future_predictions = predict_future_prices(model, scaler, last_real_data, seq_length, num_future_days)
                        
                        if len(future_predictions) > 0:
                            # Generar las fechas para las predicciones futuras
                            last_date = data.index[-1]
                            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_future_days, freq='B') # 'B' para días hábiles
                            
                            self.ax.plot(future_dates, future_predictions, label=f"Predicción Próx. {num_future_days} Días", color='cyan', linestyle='-.')
                        else:
                            print("No se pudieron generar predicciones futuras.")
                    else:
                        print(f"No hay suficientes datos ({len(data['Close'])} puntos) para formar la secuencia ({seq_length}) para predicciones futuras.")

                self.ax.set_title(f"Gráfica de Precios de {ticker_name} con Predicción LSTM ({selected_period_display})", color='white')
                self.ax.set_xlabel("Fecha", color='white')
                self.ax.set_ylabel("Precio de Cierre", color='white')
                self.ax.tick_params(axis='x', colors='white')
                self.ax.tick_params(axis='y', colors='white')
                self.ax.spines['bottom'].set_color('white')
                self.ax.spines['top'].set_color('white')
                self.ax.spines['right'].set_color('white')
                self.ax.spines['left'].set_color('white')
                self.ax.grid(True, linestyle='--', alpha=0.7)
                self.fig.autofmt_xdate()
                self.ax.legend() 
            else:
                self.ax.text(0.5, 0.5, f"No se pudieron cargar datos o entrenar el modelo para {ticker_name}",
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=self.ax.transAxes, fontsize=12, color='gray')
                self.ax.set_title(f"Gráfica de {ticker_name} (sin datos)", color='white')
        else:
            self.ax.text(0.5, 0.5, "Selecciona un ticker para ver su gráfica y predicción",
                                   horizontalalignment='center', verticalalignment='center',
                                   transform=self.ax.transAxes, fontsize=12, color='gray')
            self.ax.set_title("Gráfica Vacía", color='white')

        self.fig.tight_layout()
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = App()
    app.mainloop()
