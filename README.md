# Fidelidad Vocacional — Universidad de Antioquia

Modelos de red neuronal para predecir la probabilidad de que un estudiante admitido se gradúe del mismo programa en el que se inscribió. Trabajo final — Especialización en Analítica y Ciencia de Datos, materia Deep Learning.

## Archivos necesarios

- `graduados.csv` — registros de egresados
- `matriculados.csv` — registros de inscripciones

Los archivos se descargan automáticamente desde Google Drive al ejecutar el notebook.

## Notebook

`prediccion_graduacion.ipynb` — pipeline completo: limpieza, encoding, entrenamiento y evaluación de cuatro experimentos:

| Experimento | Arquitectura | Concepto clave |
|---|---|---|
| **Exp A** | MLP simple (82→50→30→1) | Baseline: red densa plana |
| **Exp B** | Multimodal: MLP + Embedding de programa | Representación separada para variable categórica de alta cardinalidad |
| **Exp C** | Multimodal + BatchNorm + Dropout + Residual | Regularización y conexiones residuales (skip connections) |
| **Exp D** | Transfer Learning: Autoencoder → fine-tuning | Pre-entrenamiento no supervisado + transferencia de representación |

## Reproducibilidad

Todas las semillas están fijadas globalmente (`SEED = 42`) sobre `random`, `numpy`, `tensorflow` y variables de entorno. El split de validación es fijo y el oversampling se aplica exclusivamente sobre el conjunto de entrenamiento.

## Dependencias

```
tensorflow
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
gdown
```

## Integrantes

- Camilo Valencia
- Angel Rey
