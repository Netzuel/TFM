# TFM
Repositorio para almacenar el código y demás que vaya haciendo en mi TFM. Durante el proyecto contamos con una serie de imágenes de leucoplasias de pacientes reales. Estas imágenes son preprocesadas con el objetivo de escribir un modelo de clasificación que sea capaz de decirnos, a partir de una imagen de la herida oral en cuestión, si el paciente va a desarrollar o no un cáncer a partir de dicha lesión. Como objetivo secundario se plantea la posibilidad de entrenar algún modelo de aprendizaje profundo para extraer, de forma automática, las heridas orales a partir de una imagen de la boca en su totalidad.

Pasos por ahora:
* Realizacion del preprocesado de las lesiones con éxito.
* Disponemos únicamente de 12 imágenes de la totalidad de las 307, pero se ha hecho uso de técnicas de Data Augmentation con lo que terminamos con 84.
* Se ha probado el modelo más básico, una regresión logística sin tener en cuenta la información sobre la histología, consiguiendo un AUC de aproximadamente 56%.
