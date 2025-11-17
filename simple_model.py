import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    base_path = "C:/Users/Yarik32/Storage/Python/isGameModel"
    features = [f"color{i}" for i in range(21696)]
    return base_path, features


@app.cell
def _(Image):
    def resize_image(image, new_width, new_height):
        return image.resize((new_width, new_height), Image.Resampling.BOX)
    return (resize_image,)


@app.cell
def _(Image, base_path, features, np, os, pd, resize_image):
    def load_images(images_type="work", target=True):
        images = []
        for image_name in os.listdir(f"{base_path}/{images_type}"):
            image_path = f"{base_path}/{images_type}/{image_name}"
            image = Image.open(image_path).convert("RGB")
            resized_image = resize_image(image, 113, 64)
            image_array = np.array(resized_image).astype('float32') / 255
            flat_image = image_array.flatten()
            images.append(flat_image)
        df = pd.DataFrame(images, columns=features)
        if target:
            df["game"] = images_type == "game"
        return df
    return (load_images,)


@app.cell
def _(load_images, pd):
    full_df = pd.concat([load_images(), load_images("game")])
    return (full_df,)


@app.cell
def _(features, full_df):
    features_df = full_df[features]
    target_df = full_df["game"].astype(int)
    return features_df, target_df


@app.cell
def _(features_df, target_df, train_test_split):
    train_features, test_features, train_target, test_target = train_test_split(features_df, target_df, test_size=0.3, random_state=32)
    return test_features, test_target, train_features, train_target


@app.function
def gini(roc_auc):
    return 2 * roc_auc - 1


@app.cell
def _(
    LogisticRegression,
    roc_auc_score,
    test_features,
    test_target,
    train_features,
    train_target,
):
    model = LogisticRegression(random_state=32, class_weight='balanced')
    model.fit(train_features, train_target)

    train_proba = model.predict_proba(train_features)[:, 1]

    test_proba = model.predict_proba(test_features)[:, 1]
    test_pred = model.predict(test_features)

    train_roc_auc = roc_auc_score(train_target, train_proba)
    train_gini = gini(train_roc_auc)

    test_roc_auc = roc_auc_score(test_target, test_proba)
    test_gini = gini(test_roc_auc)
    return model, test_gini, test_pred, test_roc_auc, train_gini, train_roc_auc


@app.cell(hide_code=True)
def _(
    classification_report,
    mo,
    test_gini,
    test_pred,
    test_roc_auc,
    test_target,
    train_gini,
    train_roc_auc,
):
    mo.md(rf"""
    ## Результати

    ```
    {classification_report(test_target, test_pred)}
    ```

    Тренувальна вибірка:
    - ROC AUC: {train_roc_auc:.4f}
    - Gini: {train_gini:.4f}

    Тестова вибірка:
    - ROC AUC: {test_roc_auc:.4f}
    - Gini: {test_gini:.4f}
    """)
    return


@app.cell
def _(load_images):
    new_test_df = load_images("new images", False)
    return (new_test_df,)


@app.cell
def _(model, new_test_df):
    new_test_proba = model.predict_proba(new_test_df)[:, 1]
    for el in new_test_proba:
        print(f"{el:.4f}")
    model.predict(new_test_df)
    return


@app.cell
def _(Image, mss):
    def take_screenshot():
        with mss() as sct:
            new_screenshot = sct.grab(sct.monitors[0])
        return Image.frombytes("RGB", new_screenshot.size, new_screenshot.bgra, "raw", "BGRX")
    return (take_screenshot,)


@app.cell
def _(features, np, pd, resize_image, take_screenshot):
    def prepare_screenshot():
        images = []
        image = take_screenshot()
        resized_image = resize_image(image, 113, 64)
        image_array = np.array(resized_image).astype('float32') / 255
        flat_image = image_array.flatten()
        images.append(flat_image)
        df = pd.DataFrame(images, columns=features)
        return df
    return (prepare_screenshot,)


@app.cell
def _(mo):
    live_refresh = mo.ui.refresh(["1s", "3s"], label="Evaluate desktop in real time")
    return (live_refresh,)


@app.cell(hide_code=True)
def _(live_refresh, mo, model, prepare_screenshot):
    mo.md(rf"""
    {live_refresh}

    {model.predict_proba(prepare_screenshot())[:, 1][0]:.2%}
    """)
    return


@app.cell
def _():
    from mss.windows import MSS as mss
    return (mss,)


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn import set_config

    set_config(transform_output="pandas")
    return (
        LogisticRegression,
        classification_report,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _():
    import marimo as mo
    from PIL import Image
    import os
    import numpy as np
    import pandas as pd
    return Image, mo, np, os, pd


if __name__ == "__main__":
    app.run()
