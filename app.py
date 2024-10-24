
import dash
from dash import dcc, html, callback, Output, Input
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import dash_table

# Inicializamos la app Dash
app = dash.Dash(__name__)
server = app.server

def load_data():
    # Inicializar una lista para almacenar los DataFrames
    dataframes = []
    
    # Lista de archivos a cargar
    files = [
        'data/resultado_part1.txt',
        'data/resultado_part2.txt',
        'data/resultado_part3.txt'
    ]

    for file in files:
        # Cargar los datos desde el archivo .txt
        df = {'bid_price': [], 'bid_volume': [], 'ask_price': [], 'ask_volume': []}
        
        with open(file, 'r') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if index > 1:  # Ignora las primeras dos líneas si son encabezados
                    splitted_line = line.split(',')
                    
                    # Verificar que la línea tenga suficientes elementos
                    if len(splitted_line) >= 6:  # Asegúrate de que hay al menos 6 elementos
                        try:
                            # Intentar convertir los valores y manejar errores
                            bid_price = float(splitted_line[2]) if splitted_line[2] else None
                            bid_volume = float(splitted_line[3]) if splitted_line[3] else None
                            ask_price = float(splitted_line[4]) if splitted_line[4] else None
                            ask_volume = float(splitted_line[5]) if splitted_line[5] else None
                            
                            # Solo agregar a la lista si todos los valores son válidos
                            if None not in (bid_price, bid_volume, ask_price, ask_volume):
                                df['bid_price'].append(bid_price)
                                df['bid_volume'].append(bid_volume)
                                df['ask_price'].append(ask_price)
                                df['ask_volume'].append(ask_volume)
                        except ValueError as ve:
                            print(f"Error de conversión en línea {index} de {file}: {line.strip()} - {ve}")
                    else:
                        print(f"Línea ignorada por longitud insuficiente en {file}: {line.strip()}")

        # Convertir a DataFrame y agregar a la lista si hay datos
        if df['bid_price']:  # Asegurarse de que no está vacío
            dataframes.append(pd.DataFrame(df))

    # Concatenar todos los DataFrames en uno solo si hay datos
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Preparar los datos
        X = combined_df.drop('ask_price', axis=1)
        y = combined_df['ask_price']

        # Dividir los datos
        X_train, X_test = X.iloc[:-10000], X.iloc[-10000:]
        y_train, y_test = y.iloc[:-10000], y.iloc[-10000:]

        return X_train, X_test, y_train, y_test
    else:
        raise ValueError("No se cargaron datos válidos de ninguno de los archivos.")



# Función para calcular métricas
def calcular_metricas(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {'RMSE': [rmse], 'MAPE': [mape], 'R2': [r2]}

# Layout de la aplicación Dash
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("Modelo de Machine Learning para Predicción de Precios", style={'textAlign': 'center', 'color': '#2C3E50'}),
    html.Button('Entrenar Modelos', id='train-button', style={'margin': '20px auto', 'display': 'block', 'backgroundColor': '#3498DB', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'fontSize': '16px', 'borderRadius': '5px'}),
    html.Div(id='output-div', style={'textAlign': 'center', 'fontSize': '18px', 'color': '#2C3E50'}),
    
    html.Div([
        dcc.Graph(id='graph-linear', style={'height': '40vh'}),
        dash_table.DataTable(id='table-linear', columns=[{"name": "Métrica", "id": "metric"}, {"name": "Valor", "id": "value"}], style_table={'overflowX': 'auto'}, style_header={'backgroundColor': '#3498DB', 'color': 'white'}, style_cell={'textAlign': 'center'}),
    ], style={'marginBottom': '40px'}),
    
    html.Div([
        dcc.Graph(id='graph-ridge', style={'height': '40vh'}),
        dash_table.DataTable(id='table-ridge', columns=[{"name": "Métrica", "id": "metric"}, {"name": "Valor", "id": "value"}], style_table={'overflowX': 'auto'}, style_header={'backgroundColor': '#3498DB', 'color': 'white'}, style_cell={'textAlign': 'center'}),
    ], style={'marginBottom': '40px'}),
    
    html.Div([
        dcc.Graph(id='graph-lasso', style={'height': '40vh'}),
        dash_table.DataTable(id='table-lasso', columns=[{"name": "Métrica", "id": "metric"}, {"name": "Valor", "id": "value"}], style_table={'overflowX': 'auto'}, style_header={'backgroundColor': '#3498DB', 'color': 'white'}, style_cell={'textAlign': 'center'}),
    ], style={'marginBottom': '40px'}),
    
    html.Div([
        dcc.Graph(id='graph-knn', style={'height': '40vh'}),
        dash_table.DataTable(id='table-knn', columns=[{"name": "Métrica", "id": "metric"}, {"name": "Valor", "id": "value"}], style_table={'overflowX': 'auto'}, style_header={'backgroundColor': '#3498DB', 'color': 'white'}, style_cell={'textAlign': 'center'}),
    ], style={'marginBottom': '40px'}),
])

# Callback para entrenar el modelo y actualizar las gráficas
@app.callback(
    Output('output-div', 'children'),
    Output('graph-linear', 'figure'),
    Output('table-linear', 'data'),
    Output('graph-ridge', 'figure'),
    Output('table-ridge', 'data'),
    Output('graph-lasso', 'figure'),
    Output('table-lasso', 'data'),
    Output('graph-knn', 'figure'),
    Output('table-knn', 'data'),
    Input('train-button', 'n_clicks')
)
def update_output(n_clicks):
    if n_clicks is None:
        return "Presiona el botón para entrenar los modelos.", go.Figure(), [], go.Figure(), [], go.Figure(), [], go.Figure(), []

    # Cargar datos
    X_train, X_test, y_train, y_test = load_data()
    
    # Entrenar modelos
    models = {
        'Lineal': Pipeline([('scaler', MinMaxScaler()), ('classifier', LinearRegression())]),
        'Ridge': Pipeline([('scaler', MinMaxScaler()), ('classifier', Ridge(alpha=1))]),
        'Lasso': Pipeline([('scaler', MinMaxScaler()), ('classifier', Lasso(alpha=1))]),
        'KNN': Pipeline([('scaler', MinMaxScaler()), ('classifier', KNeighborsRegressor(n_neighbors=5))])
    }
    
    y_preds = {}
    metrics = {}

    # Entrenamiento y predicción para cada modelo
    for name, model in models.items():
        model.fit(X_train, y_train)  # Entrenar el modelo
        y_pred = model.predict(X_test)  # Hacer predicciones
        y_preds[name] = y_pred
        metrics[name] = calcular_metricas(y_test, y_pred)  # Calcular métricas

    # Crear gráfica y tabla para cada modelo
    figures = {}
    tables = {}

    for name, y_pred in y_preds.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='Valores Reales'))
        fig.add_trace(go.Scatter(x=np.arange(len(y_pred)), y=y_pred, mode='lines', name=f'Predicciones {name}'))

        fig.update_layout(
            title=f'Resultados de Predicción - {name}',
            xaxis_title='Observaciones',
            yaxis_title='Precio'
        )

        figures[name] = fig
        # Crear datos para la tabla
        tables[name] = [
            {"metric": "RMSE", "value": metrics[name]['RMSE'][0]},
            {"metric": "MAPE", "value": metrics[name]['MAPE'][0]},
            {"metric": "R2", "value": metrics[name]['R2'][0]},
        ]

    return (
        f"Modelos entrenados con éxito.",
        figures['Lineal'], tables['Lineal'],
        figures['Ridge'], tables['Ridge'],
        figures['Lasso'], tables['Lasso'],
        figures['KNN'], tables['KNN']
    )

# Ejecutar la app
if __name__ == '__main__':
    app.run_server(debug=True)











