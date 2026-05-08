import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.manifold import TSNE


def plot_learning_curves(histories, labels, title_suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for hist, label in zip(histories, labels):
        ax1.plot(hist['val_acc'], label=label, marker='o')
        ax2.plot(hist['train_loss'], label=f'Train {label}', linestyle='--')
        ax2.plot(hist['val_loss'], label=f'Val {label}', marker='x')

    ax1.set_title(f'Accuracy na zbiorze walidacyjnym {title_suffix}')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title(f'Funkcja straty (Loss) {title_suffix}')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Prawdziwa klasa')
    plt.title('Macierz Błędów (Confusion Matrix)')
    plt.show()


def plot_misclassified(images, y_true, y_pred, classes, num_images=5):
    misclassified_idx = np.where(np.array(y_true) != np.array(y_pred))[0]

    plt.figure(figsize=(15, 3))
    for i in range(min(num_images, len(misclassified_idx))):
        idx = misclassified_idx[i]
        img = images[idx]
        img = img / 2 + 0.5
        img = np.transpose(img, (1, 2, 0))

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(f'P: {classes[y_pred[idx]]}\nO: {classes[y_true[idx]]}', color='red')
        plt.axis('off')
    plt.tight_layout()
    plt.show()