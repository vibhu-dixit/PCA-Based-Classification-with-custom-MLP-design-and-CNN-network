import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def create_model(learning_rate=0.001, kernel_size=(3, 3), batch_norm=True, dropout_rates=(0.2, 0.3, 0.4)):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size, padding='same', input_shape=(32, 32, 3))) # First Convolutional Layer
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3, 3), padding='same')) # Second Convolutional Layer
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2, 2))) # Max Pooling
    
    model.add(Dropout(dropout_rates[0])) # Dropout Layer
    
    model.add(Conv2D(64, (3, 3), padding='same')) # Third Convolutional Layer
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3, 3), padding='same')) # Fourth Convolutional Layer
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2, 2))) # Max Pooling
    
    model.add(Dropout(dropout_rates[1])) # Dropout Layer
    
    model.add(Conv2D(128, (3, 3), padding='same'))# Fifth Convolutional Layer
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())
    
    model.add(Conv2D(128, (3, 3), padding='same'))# Sixth Convolutional Layer
    model.add(Activation('relu'))
    if batch_norm:
        model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2, 2))) # Max Pooling
    
    model.add(Dropout(dropout_rates[2])) # Dropout Layer

    model.add(Flatten())# Flatten and Fully Connected Layer
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def experiment_with_hyperparameters():
    results = {}

    print("\nTraining base model")
    model = create_model()
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results["Base Model"] = test_acc
    plot_loss_accuracy(history, "Base Model")

    for lr in [0.05, 0.0001]:
        print(f"\nTraining with learning rate: {lr}")
        model = create_model(learning_rate=lr)
        history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=2)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[f"LR={lr}"] = test_acc
        plot_loss_accuracy(history, f"Learning Rate = {lr}")

    print("\nTraining with 7x7 kernel size for first layer")
    model = create_model(kernel_size=(7, 7))
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results["Kernel=7x7"] = test_acc
    plot_loss_accuracy(history, "Kernel Size = 7x7")

    print("\nTraining with no batch normalization layers")
    model = create_model(batch_norm=False)
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results["No Batch Norm"] = test_acc
    plot_loss_accuracy(history, "No Batch Norm")

    for batch_size in [16, 256]:
        print(f"\nTraining with batch size: {batch_size}")
        model = create_model()
        history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[f"Batch Size={batch_size}"] = test_acc
        plot_loss_accuracy(history, f"Batch Size = {batch_size}")

    return results

def plot_loss_accuracy(history, experiment_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{experiment_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{experiment_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(f'{experiment_name}_loss_accuracy.png')
    plt.show()

results = experiment_with_hyperparameters()
print("\nExperiment Results: Test Accuracy for Different Settings")
for setting, accuracy in results.items():
    print(f"{setting}: {accuracy:.4f}")
