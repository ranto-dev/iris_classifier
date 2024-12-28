import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
loaded_model = tf.keras.models.load_model('./model.h5')

# Définir les classes
CLASSES = np.array(["Avion", "Automobile", "Oiseau", "Chat", "Cerf", "Chien", "Grenouille", "Cheval", "Bateau", "Camion"])

# Prédire les classes pour les données de test
predictions = loaded_model.predict(x_test)
predictions_simple = CLASSES[np.argmax(predictions, axis=-1)]
vraies_classes_simples = CLASSES[np.argmax(y_test, axis=-1)]


# Nombre d'images à afficher
n_image = 6
indices = np.random.choice(range(len(x_test)), n_image)

# Créer une figure pour afficher les images
fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# Afficher les images avec leurs prédictions
for i, idx in enumerate(indices):
    image = x_test[idx]
    ax = fig.add_subplot(1, n_image, i + 1)
    ax.axis("off")
    ax.text(
        0.5,
        -0.35,
        "Prédiction = " + str(predictions_simple[idx]),
        fontsize = 10,
        ha = "center",
        transform = ax.transAxes,
    )
    ax.text(
        0.5,
        -0.7,
        "Etiquette réelle = " + str(vraies_classes_simples[idx]),
        fontsize = 10,
        ha = "center",
        transform = ax.transAxes,
    )
    ax.imshow(image)

# Afficher la figure
plt.show()