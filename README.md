# Visión Artificial para Clasificación de Residuos

## Descripción

Este proyecto utiliza un modelo de visión artificial basado en MobileNet v3 para la clasificación de residuos.

El objetivo es desarrollar un sistema que pueda identificar y clasificar diferentes tipos de residuos utilizando imagenes capturadas por una camara.

El modelo clasifica las imagenes en una de las siguientes categorias:
* carboard
* glass
* metal
* paper
* plastic
* trash

## Estructura del Proyecto

* 'customdata' - Directorio con imagenes tomadas desde celular para probar el modelo
* 'models' - Modelos Entrenados
* 'modules' - Scripts principales para el entrenamiento del modelo
* 'notebooks' - Notebooks de testing con código sin modularizar

## Requisitos

Para ejecutar este proyecto, necesitas instalar las siguientes bibliotecas:
1. numpy
2. pandas
3. torch
4. torchvision
5. Pillow

## Instalación

Puedes instalar estas dependencias utilizando el archivo *'requirements.txt'*

1. Clona el repositorio:
2. Navega al directorio del proyecto
3. Instala las dependencias:

## Uso

### Entrenamiento del Modelo

### Evaluación del Modelo

### Prueba del Modelo
Para probar el modelo con nuevas imágenes:
1. Coloca las imagenes en el directorio '/customdata'
2. Ejecuta el script de prueba (completar)

## Contribuciones

## Licencia

## Contacto

## Agradecimientos

Para entrenar este modelo me basé en lo aprendido en el curso: https://github.com/mrdbourke/pytorch-deep-learning

Por otro lado, para seleccionar el dataset me baso en el review: https://github.com/AgaMiko/waste-datasets-review

En el proyecto utilizo trashnet dataset: https://github.com/garythung/trashnet
