import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocessing import preprocess
from model import build_unetplusplus

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true * y_pred)
    
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )
    
def iou(y_true, y_pred):
    smooth = 1e-6
    
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    
    return (intersection + smooth) / (union + smooth)

if __name__ == "__main__":

    print("===== U-Net++ Segmentation =====")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    image_dir = os.path.join(base_dir, "dataset/images")
    mask_dir = os.path.join(base_dir, "dataset/annotations/trimaps")

    # 🔹 Step 1: Load Data
    print("Loading data...")
    images, masks = load_data(image_dir, mask_dir)
    print("Loaded:", len(images))

    # 🔹 Step 2: Preprocess
    print("Preprocessing...")
    X, Y = preprocess(images, masks)
    print("X:", X.shape, "Y:", Y.shape)

    # 🔹 Step 3: Split
    print("Splitting data...")
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)

    # 🔹 Step 4: Build Model
    print("Building model...")
    model = build_unetplusplus()
    model.summary()

    # 🔹 Step 5: Compile
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss=combined_loss,
        metrics=[dice_coef,iou]
    )

    # 🔹 Step 6: Train
    print("Training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=10,
        batch_size=8,verbose=1)
    

    # 🔹 Step 7: Save Model
    model.save("unetplusplus_model.h5")
    print("Model saved!")

    # 🔹 Step 8: Predict
    print("Predicting...")
    preds = model.predict(X_val[:3])

    # 🔹 Step 9: Visualization
    for i in range(3):
        plt.figure(figsize=(10,3))

        plt.subplot(1,3,1)
        plt.title("Image")
        plt.imshow(X_val[i])

        plt.subplot(1,3,2)
        plt.title("True Mask")
        plt.imshow(Y_val[i].squeeze(), cmap='gray')

        plt.subplot(1,3,3)
        plt.title("Predicted Mask")
        plt.imshow(preds[i].squeeze() > 0.5, cmap='gray')

        plt.show()

    # 🔹 Step 10: Evaluation
    loss, dice, iou_score = model.evaluate(X_val, Y_val)
    print("Validation Loss:", loss)
    print("Dice Score:", dice)
    print("IoU Score:", iou_score)