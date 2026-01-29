# Sistema de Telediagnóstico Asistido por IA para Detección de Cáncer de Mama

## Descripción del Proyecto
Este proyecto consiste en un sistema de telediagnóstico basado en Deep Learning diseñado para la clasificación de imágenes histopatológicas de cáncer de mama. El objetivo principal es reducir las brechas de diagnóstico en zonas rurales de la Región de Arica y Parinacota, proporcionando una herramienta de triaje inteligente que priorice casos con alta sospecha de malignidad.

## Arquitectura del Modelo
El motor de clasificación utiliza una Red Neuronal Convolucional (CNN) basada en la arquitectura **ResNet50**.
* **Arquitectura**: ResNet50 implementada mediante la librería `timm`.
* **Técnica**: Transfer Learning sobre el dataset BreaKHis (Breast Cancer Histopathological Database).
* **Clasificación**: Binaria (Benigno vs. Maligno).
* **Optimizador**: Adam con una tasa de aprendizaje de 1e-4.
* **Función de Pérdida**: CrossEntropyLoss.



## Estructura del Código

### 1. Entrenamiento y Validación (Notebook)
El proceso de desarrollo se documentó en un entorno de notebook (`prueba.ipynb`) donde se ejecutaron las siguientes etapas:
* **Pre-procesamiento de Datos**: Implementación de una función de aplanado (*flatten*) para organizar las imágenes del dataset original en directorios de entrenamiento y validación.
* **Entrenamiento**: Ejecución de 5 épocas utilizando una GPU NVIDIA GeForce RTX 2060, logrando reducir la pérdida de 0.5189 a 0.1048.
* **Evaluación**: Generación de métricas de desempeño con una precisión (accuracy) final del 98%.

### 2. Aplicación y Despliegue (Streamlit)
La interfaz de usuario se desarrolló con **Streamlit**, permitiendo el uso del modelo de forma interactiva en la nube.
* **Carga Eficiente**: Uso de `@st.cache_resource` para mantener el modelo cargado en memoria y optimizar el tiempo de respuesta.
* **Inferencia**: Las imágenes subidas por el usuario se transforman a tensores de 224x224 píxeles y se normalizan antes de pasar por el modelo.
* **Lógica de Resultados**: El sistema aplica una función Softmax; si la probabilidad de malignidad es superior al 50%, emite una alerta de prioridad alta para derivación al especialista.

## Despliegue Temporal
El despliegue del prototipo se gestionó de la siguiente manera:
1. **GitHub**: Alojamiento del código fuente (`app.py`) y control de versiones.
2. **Streamlit Cloud**: Conexión directa al repositorio para servir la aplicación web de forma gratuita y accesible mediante navegador.
3. **Persistencia**: El archivo `modelo_cancer.pth` contiene los pesos finales del modelo entrenado y debe estar presente en el directorio raíz para el funcionamiento de la aplicación.

## Requisitos del Sistema
Para replicar el entorno localmente, se requiere Python 3.11+ y las siguientes librerías:
* torch
* torchvision
* timm
* streamlit
* pillow
* scikit-learn

## Instrucciones de Uso
1. Clonar el repositorio.
2. Instalar dependencias mediante `pip install -r requirements.txt`.
3. Ejecutar la aplicación con el comando `streamlit run app.py`.
4. Subir una imagen de histopatología teñida con H&E para obtener el pre-diagnóstico.