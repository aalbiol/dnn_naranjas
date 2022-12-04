<h1><center>ResNet-MIL</center></h1> 
Permite entrenar un modelo Resnet para clasificar vistas de naranjas en 3 categorías, basándose únicamente en la anotación a nivel de fruto empleando MIL

En el entrenamiento de cada, fruto emplea la vista cuya probabilidad de defecto sea mayor.

Para entrenar se ha hecho:

  * 5000 epochs refinando solo la capa fully connected de resnet
  
  * Partiendo de lo anterior, otros 5000 epochs fine-tuning todos los pesos

Se puede elegir entre distintas Resnets (probado con 18)

Se ha entrenado con SGD
  

Ejemplos de uso para entrenar y predecir: 

```
python train_3clases -g 1 -to --num_epochs 5000

python train_3clases -g 1  --num_epochs 2000 --initial_model modelo_partida.ckpt

python predict_3clases -d directorio_imagenes -m modelo.ckpt -g 1


```
