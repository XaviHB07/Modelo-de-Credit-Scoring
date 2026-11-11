# Modelo de Credit Scoring

Modelo para estimar la probabilidad de default de emprendedores que obtuvieron un préstamo personal o de micro y pequeña empresa en febrero (2018 para entrenar, 2019 para evaluar), con los datos de la Datathon Interbank 2020. La métrica de la competencia es el AUC.

Se usan cuatro fuentes: el historial en el RCC, la información de SUNAT, las variables sociodemográficas (SE) y las del Censo 2017. Del RCC se arman variables por cliente con ventanas de 3, 6 y 12 meses previas al desembolso (deuda y su tendencia, instituciones y productos, clasificación del deudor y días de mora, composición por tipo de producto y uso de la línea de tarjeta); el mes del desembolso queda fuera. SUNAT se resume a un registro por cliente, SE y Censo se limpian y se podan por correlación, y el Censo se resume además con PCA. Los códigos con muchos niveles (ubigeo, ciiu, etc.) se codifican con target encoding dentro de cada fold.

Se compararon regresión logística, random forest y LightGBM (con Optuna), y el modelo final es un blend de LightGBM y logística calibrado con Platt. Se validó con un holdout del 20% y validación cruzada de 5 folds. Como la cohorte 2019 no tiene etiquetas, se midió el desfase con 2018 con una validación adversarial y se quitaron las 15 variables que más delataban la cohorte.

En el holdout el AUC es 0,838 (IC 95%: 0,824–0,849), con KS de 0,50 y un lift de 4,3 en el decil de mayor riesgo. En validación cruzada la logística da 0,819, el random forest 0,816 y LightGBM 0,833. El RCC explica cerca del 73% de la ganancia del modelo, el Censo 16%, SE 6% y SUNAT 5%. Las métricas salen de una submuestra de 40 mil clientes (32 mil para ajustar y 8 mil de holdout) y el modelo final se entrenó con 150 mil; en `reports/` están las métricas, las figuras y el `submission.csv`.

## Datos

Para facilitar la descarga y el uso del repositorio, se incluyen los datasets procesados en formato Parquet (`.parquet`), que permiten almacenar y trabajar con los datos de manera más eficiente y ocupando menos espacio.

Los archivos originales (`.csv`) no se incluyen en este repositorio debido a su tamaño. Se pueden descargar directamente desde la competencia original:

https://www.kaggle.com/competitions/interbank20

Los archivos originales deben colocarse en `data/raw/`, mientras que los archivos procesados se encuentran en `data/processed/`.

## Ejecución

Para ejecutar el proyecto hay que instalar las dependencias:
```
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Si se desea reproducir todo el procesamiento desde cero, se deben descargar los archivos originales de Kaggle, colocarlos en data/raw/ y ejecutar los notebooks en orden, del 00 al 05, o bien el pipeline por consola: python scripts/run_pipeline.py
