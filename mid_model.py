import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Результати простої моделі

    ```
                  precision    recall  f1-score   support

               0       0.91      0.97      0.94        75
               1       0.97      0.90      0.93        68

        accuracy                           0.94       143
       macro avg       0.94      0.94      0.94       143
    weighted avg       0.94      0.94      0.94       143
    ```

    Тренувальна вибірка:
    - ROC AUC: 1.0000
    - Gini: 1.0000

    Тестова вибірка:
    - ROC AUC: 0.9910
    - Gini: 0.9820
    """)
    return


@app.cell(hide_code=True)
def _(
    class_names,
    classification_report,
    gini,
    mo,
    roc_auc,
    y_pred_binary,
    y_true,
):
    mo.md(rf"""
    ## Результати CNN моделі

    ```
    {classification_report(y_true, y_pred_binary, target_names=class_names)}
    ```

    Тестова вибірка:
    - ROC AUC: {roc_auc:.4f}
    - Gini: {gini:.4f}
    """)
    return


@app.cell
def _(tf):
    inference_model = tf.keras.models.load_model('desktop_classifier.keras')
    return (inference_model,)


@app.cell
def _():
    DATA_DIR = 'dataset'
    BATCH_SIZE = 32
    IMG_HEIGHT = 180
    IMG_WIDTH = 180
    EPOCHS = 20
    SEED = 591
    return BATCH_SIZE, DATA_DIR, EPOCHS, IMG_HEIGHT, IMG_WIDTH, SEED


@app.cell
def _(BATCH_SIZE, DATA_DIR, IMG_HEIGHT, IMG_WIDTH, SEED, layers, tf):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.3,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.3,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # Keep data in memory for faster CPU access
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
      layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.1),
    ])
    return class_names, data_augmentation, train_ds, val_ds


@app.cell(disabled=True)
def _(EPOCHS, data_augmentation, layers, models, tf, train_ds, val_ds):
    model = models.Sequential([
        data_augmentation,

        layers.Rescaling(1./255),

        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Dropout(0.2), 

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )
    return history, model


@app.cell
def _(model):
    model.save('desktop_classifier.keras')
    return


@app.cell
def _(history, plt):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.show()
    return


@app.cell
def _(
    class_names,
    classification_report,
    inference_model,
    np,
    plt,
    roc_auc_score,
    roc_curve,
    val_ds,
):
    y_true = []
    y_pred_probs = []

    for images, labels in val_ds:
        y_true.extend(labels.numpy())
        preds = inference_model.predict(images, verbose=0)
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    print("\n" + "="*30)
    print("CNN MODEL PERFORMANCE")
    print("="*30)

    print(classification_report(y_true, y_pred_binary, target_names=class_names))

    roc_auc = roc_auc_score(y_true, y_pred_probs)
    gini = 2 * roc_auc - 1

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Gini:    {gini:.4f}")

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (CNN)')
    plt.legend(loc="lower right")
    plt.show()
    return gini, roc_auc, y_pred_binary, y_true


@app.cell
def _(inference_model, tf):
    def new_image(img_path):
        img = tf.keras.utils.load_img(
            img_path, target_size=(180, 180)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = inference_model.predict(img_array)
        score = float(predictions[0][0])

        print(img_path)
        print(f"Raw Score: {score}")
        if score > 0.5:
            print(f"WORK {100 * score:.2f}%")
        else:
            print(f"GAME {100 * (1 - score):.2f}%")
    return (new_image,)


@app.cell
def _(new_image, os):
    for image_name in os.listdir("new images"):
        new_image(f"new images/{image_name}")
    return


@app.cell(hide_code=True)
def _(inference_model, live_refresh, mo, take_screenshot_for_model):
    mo.md(rf"""
    {live_refresh}

    {float(inference_model.predict(take_screenshot_for_model(), verbose=0)[0][0]):.2%}
    """)
    return


@app.cell
def _(IMG_HEIGHT, IMG_WIDTH, Image, mss, tf):
    def take_screenshot_for_model():
        with mss() as sct:
            new_screenshot = sct.grab(sct.monitors[0])
        img = Image.frombytes("RGB", new_screenshot.size, new_screenshot.bgra, "raw", "BGRX")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array
    return (take_screenshot_for_model,)


@app.cell
def _(mo):
    live_refresh = mo.ui.refresh(["1s", "3s"], label="Evaluate desktop in real time")
    return (live_refresh,)


@app.cell
def _():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
    from mss.windows import MSS as mss
    from PIL import Image
    import marimo as mo
    return (
        Image,
        classification_report,
        layers,
        mo,
        models,
        mss,
        np,
        os,
        plt,
        roc_auc_score,
        roc_curve,
        tf,
    )


if __name__ == "__main__":
    app.run()
