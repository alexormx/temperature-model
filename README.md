
# Predicción de Temperatura con Regresión Lineal

Este proyecto utiliza un modelo de regresión lineal para predecir la temperatura del siguiente día basado en los últimos 10 días de registros de temperatura. El modelo utiliza Python y bibliotecas comunes de machine learning.

---

## **Requisitos del proyecto**

Antes de ejecutar este proyecto, asegúrate de tener instalado lo siguiente:

- Python 3.8 o superior
- Las siguientes bibliotecas de Python:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib` (opcional, para visualización)

Para instalar las dependencias, ejecuta el siguiente comando:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## **Estructura del proyecto**

```
.
├── data/
│   └── dataset.csv     # Archivo con los registros históricos de temperatura
├── src/
│   ├── train_model.py  # Script principal para entrenar el modelo
│   ├── predict.py      # Script para realizar predicciones
│   └── utils.py        # Funciones auxiliares
├── README.md           # Archivo de documentación
└── requirements.txt    # Dependencias del proyecto
```

---

## **Uso**

### 1. **Preparación de los datos**
Asegúrate de que el archivo `dataset.csv` en la carpeta `data` contenga una columna llamada `temperatura` con las temperaturas diarias en formato cronológico.

Ejemplo de contenido del archivo:

| temperatura |
|-------------|
| 23.5        |
| 24.1        |
| 22.8        |
| 25.0        |
| ...         |

### 2. **Entrenamiento del modelo**
Ejecuta el script `train_model.py` para entrenar el modelo con el conjunto de datos.

```bash
python src/train_model.py
```

Este script:

- Carga el conjunto de datos desde `data/dataset.csv`.
- Genera características basadas en ventanas deslizantes de los últimos 10 días.
- Entrena un modelo de regresión lineal.
- Guarda el modelo entrenado como un archivo `.pkl` para uso posterior.

### 3. **Realizar una predicción**
Para predecir la temperatura del siguiente día utilizando los últimos 10 días, ejecuta el script `predict.py`:

```bash
python src/predict.py
```

El resultado será similar a:

```plaintext
Temperatura predicha para el siguiente día: 24.87
```

---

## **Detalles técnicos**

### **Generación de características**
El modelo utiliza una ventana deslizante de 10 días consecutivos para predecir la temperatura del día siguiente. Cada ventana se construye como un vector de entrada \( X \), mientras que la salida \( y \) es la temperatura del día 11.

### **Entrenamiento del modelo**
Se utiliza un modelo de regresión lineal proporcionado por `scikit-learn`. La métrica utilizada para evaluar el modelo es el coeficiente de determinación \( R^2 \).

### **Evaluación**
Durante el entrenamiento, los datos se dividen en conjuntos de entrenamiento (80%) y prueba (20%). El modelo se evalúa utilizando el conjunto de prueba.

---

## **Ejemplo de código**

### **Entrenamiento del modelo**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar los datos
data = pd.read_csv("data/dataset.csv")

# Generar ventanas deslizantes
n_steps = 10
X, y = [], []
for i in range(len(data) - n_steps):
    X.append(data['temperatura'].iloc[i:i + n_steps].values)
    y.append(data['temperatura'].iloc[i + n_steps])

X = np.array(X)
y = np.array(y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Guardar el modelo entrenado (opcional)
import joblib
joblib.dump(model, "src/model.pkl")
```

### **Predicción**

```python
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo entrenado
model = joblib.load("src/model.pkl")

# Cargar los últimos 10 días de temperatura
data = pd.read_csv("data/dataset.csv")
last_10_days = data['temperatura'].iloc[-10:].values.reshape(1, -1)

# Realizar la predicción
next_day_temperature = model.predict(last_10_days)
print(f"Temperatura predicha para el siguiente día: {next_day_temperature[0]:.2f}")
```

---

## **Contribuciones**

Si deseas contribuir a este proyecto, por favor:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz un commit (`git commit -m 'Añadir nueva funcionalidad'`).
4. Haz un push a la rama (`git push origin feature/nueva-funcionalidad`).
5. Abre un pull request.

---

## **Licencia**

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.

---

## **Autor**

Alejandro Ortiz López  
[LinkedIn](https://www.linkedin.com/in/alexormx/) | [GitHub](https://github.com/alexormx)
