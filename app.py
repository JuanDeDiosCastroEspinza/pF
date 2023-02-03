from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        Casos_por_semana = float(request.form['Casos_por_semana'])
        Prueba_Data = pd.read_csv('Sem_H2.csv')
        X = Prueba_Data[['Cantidad de huevecillos por semana']]
        y = Prueba_Data[['Casos_por_semana']]
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict([[Casos_por_semana]])
        Casos_por_semana = y_pred[0][0]
        return render_template('index.html', Casos_por_semana=Casos_por_semana, plot='')

    else:
        Prueba_Data = pd.read_csv('Sem_H2.csv')
        X = Prueba_Data[['Cantidad de huevecillos por semana']]
        y = Prueba_Data[['Casos_por_semana']]
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)

        plt.scatter(X, y, color='red')
        plt.plot(X, y_pred, color='blue')
        plt.xlabel('Cantidad de huevecillos por semana')
        plt.ylabel('Casos_por_semana')
        plt.title('Regresión lineal')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        plot = base64.b64encode(plot_data).decode()
        return render_template('index.html', plot=plot)


if __name__ == '__main__':
    app.run(debug=True)
