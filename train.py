"""Train a 3D-CNN that classifies lip-movement video clips into 10 words (MIRACL-VC1).

Usage:
    python train.py --data /path/to/dataset
where /path/to/dataset holds one folder per speaker (F01, F02, ...), each with
words/<word 01..10>/<repeat 01..10>/color_NNN.jpg.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from keras import utils
from keras.layers import Conv3D, Dense, Flatten, Input, MaxPooling3D
from keras.models import Sequential
from PIL import Image

SEED = 42
RESULTS_DIR = "results"
FRAMES = 10  # every clip is reduced to this many frames so clips can be stacked
SIZE = 64
WORDS = ["begin", "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "web"]


def pick_frames(n):
    # n evenly spaced positions from the first to the last frame, in order.
    return np.round(np.linspace(0, n - 1, FRAMES)).astype(int)


def load_clip(folder):
    # Each folder also holds depth_NNN.png files; only the color frames are used.
    color_files = sorted(f for f in os.listdir(folder) if f.startswith("color"))
    frames = []
    for i in pick_frames(len(color_files)):
        image = Image.open(os.path.join(folder, color_files[i])).resize((SIZE, SIZE))
        frames.append(np.array(image))
    return np.stack(frames)  # shape (FRAMES, SIZE, SIZE, 3)


def load_dataset(root):
    clips, labels = [], []
    for speaker in sorted(os.listdir(root)):
        words_dir = os.path.join(root, speaker, "words")
        if not os.path.isdir(words_dir):
            continue
        for word in sorted(os.listdir(words_dir)):
            word_dir = os.path.join(words_dir, word)
            for repeat in sorted(os.listdir(word_dir)):
                clips.append(load_clip(os.path.join(word_dir, repeat)))
                labels.append(int(word) - 1)  # folder 01 to 10 becomes class 0 to 9
    return np.stack(clips), np.array(labels)


def build_model():
    model = Sequential()
    model.add(Input(shape=(FRAMES, SIZE, SIZE, 3)))

    # Pooling is (1, 3, 3): it shrinks height and width but leaves the frame axis alone.
    for filters in (16, 32, 64):
        model.add(Conv3D(filters, (3, 3, 3), activation="relu"))
        model.add(MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(10))
    model.add(Dense(10, activation="softmax"))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="folder with the speaker folders (F01, F02, ...)")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    utils.set_random_seed(SEED)

    clips, labels = load_dataset(args.data)
    print("clips", clips.shape, "labels", labels.shape)

    # Pixels run 0 to 255; scale to 0 to 1.
    clips = clips.astype("float32") / 255
    labels_onehot = utils.to_categorical(labels, 10)

    # Shuffle clips and labels together, then 1,200 for training and 300 for the final test.
    # The split is by clip, so the same speakers appear in both parts.
    order = np.random.permutation(len(clips))
    clips = clips[order]
    labels_onehot = labels_onehot[order]
    train_x, train_y = clips[:1200], labels_onehot[:1200]
    test_x, test_y = clips[1200:], labels_onehot[1200:]

    model = build_model()
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

    history = model.fit(train_x, train_y, validation_split=0.2, batch_size=32, epochs=args.epochs)
    test_loss, test_accuracy = model.evaluate(test_x, test_y)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Keeps the architecture, weights and optimizer state, so the model can predict or resume training.
    model.save(os.path.join(RESULTS_DIR, "model.keras"))

    for metric, title in (("loss", "Loss"), ("accuracy", "Accuracy")):
        plt.figure()
        plt.plot(history.history[metric], label=f"train {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val {metric}")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}.png"), dpi=120)

    # Predicted vs true word for the first 10 test clips.
    probabilities = model.predict(test_x[:10])
    lines = []
    for i in range(10):
        predicted = WORDS[int(np.argmax(probabilities[i]))]
        true = WORDS[int(np.argmax(test_y[i]))]
        lines.append(f"{i}  predicted: {predicted:<11} true: {true}")
    with open(os.path.join(RESULTS_DIR, "predictions.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    summary = {
        "parameters": model.count_params(),
        "epochs": args.epochs,
        "train_clips": 960,
        "validation_clips": 240,
        "test_clips": len(test_x),
        "final_train_accuracy": history.history["accuracy"][-1],
        "final_val_accuracy": history.history["val_accuracy"][-1],
        "test_accuracy": test_accuracy,
        "final_train_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1],
        "test_loss": test_loss,
        "best_val_loss": min(history.history["val_loss"]),
        "best_val_loss_epoch": int(np.argmin(history.history["val_loss"])) + 1,
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
