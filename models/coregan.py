# models/coregan.py

from __future__ import print_function, division
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os

from tensorflow.keras.layers import (Input, Conv2D, ConvLSTM2D, BatchNormalization,
                                     MaxPooling2D, Flatten, Dense)
from tensorflow.keras.models import Sequential, Model

class CRGAN:
    def __init__(self,
                 input_shape,
                 output_shape,
                 epochs=1000,
                 batch_size=5,
                 fact_matching_cnt=1,
                 save_dir="./results_coregan"):
        """
        input_shape: (time_step, H, W, C)
        output_shape: (H, W, C)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.fact_matching_cnt = fact_matching_cnt

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "csv"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "pred"), exist_ok=True)
        self.save_dir = save_dir

        optimizer = tf.keras.optimizers.RMSprop()
        loss_function = "binary_crossentropy"

        # build generator with fact-matching
        self.generator = self.build_generator_FM()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=loss_function, optimizer=optimizer)

        previous_image = Input(shape=self.input_shape)
        prediction_image = self.generator(previous_image)
        validity, feature_validity = self.discriminator(prediction_image)

        self.discriminator.trainable = False

        self.combined = Model(inputs=[previous_image],
                              outputs=[prediction_image, feature_validity])
        self.combined.compile(
            loss=["mse", loss_function],
            loss_weights=[1, 1],
            optimizer=optimizer,
        )

    # ---------------- Generator ----------------
    def build_generator(self):
        model = Sequential()
        model.add(
            ConvLSTM2D(
                filters=40,
                kernel_size=(3, 3),
                input_shape=self.input_shape,
                padding="same",
                return_sequences=True,
            )
        )
        model.add(BatchNormalization())
        model.add(
            ConvLSTM2D(
                filters=40,
                kernel_size=(2, 2),
                padding="same",
                return_sequences=True,
            )
        )
        model.add(BatchNormalization())
        model.add(
            ConvLSTM2D(
                filters=40,
                kernel_size=(2, 2),
                padding="same",
                return_sequences=True,
            )
        )
        model.add(BatchNormalization())
        model.add(
            ConvLSTM2D(
                filters=40,
                kernel_size=(2, 2),
                padding="same",
            )
        )
        model.add(BatchNormalization())
        model.add(
            Conv2D(
                filters=1,
                kernel_size=(3, 3),
                activation="sigmoid",
                padding="same",
                data_format="channels_last",
            )
        )
        model.summary()

        previous_steps = Input(shape=self.input_shape)
        next_step = model(previous_steps)
        return Model(previous_steps, next_step)

    def build_generator_FM(self):
        generator = self.build_generator()
        generator.compile(loss="mse", optimizer="adamax")
        return generator

    # ---------------- Discriminator ----------------
    def build_discriminator(self):
        img = Input(shape=self.output_shape)

        x = Conv2D(64, kernel_size=(2, 2), activation="relu")(img)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (2, 2), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        feat = Dense(128, activation="relu")(x)
        validity = Dense(1, activation="sigmoid")(feat)

        model = Model(img, [validity, feat])
        model.summary()
        return model

    # ---------------- Training using numpy arrays ----------------
    def train(self, x_train, y_train, x_test, y_test):
        """
        x_train: (N_train, time_step, H, W, C)
        y_train: (N_train, H, W, C)
        x_test:  (N_test, time_step, H, W, C)
        y_test:  (N_test, H, W, C)
        """
        BATCH_SIZE = self.batch_size
        epochs = self.epochs
        fact_matching_cnt = self.fact_matching_cnt

        valid = np.ones((BATCH_SIZE, 1))
        fake = np.zeros((BATCH_SIZE, 1))

        g_loss_ary = []
        d_loss_ary = []
        fm_loss_ary = []
        ff_loss_ary = []

        print("[COREGAN] x_train:", x_train.shape, "y_train:", y_train.shape)

        start_train_time = time.time()

        for ep in range(epochs):
            epoch_g_loss = []
            epoch_d_loss = []
            epoch_fm_loss = []
            epoch_ff_loss = []

            # iterate in mini-batches
            for idx in range(0, x_train.shape[0], BATCH_SIZE):
                image = x_train[idx : idx + BATCH_SIZE]
                real_image = y_train[idx : idx + BATCH_SIZE]

                if image.shape[0] < BATCH_SIZE:
                    # drop last small batch
                    continue

                # 1) generator creates next images
                gen_next_step = self.generator.predict(image, verbose=0)

                # 2) discriminator training
                d_loss_real = self.discriminator.train_on_batch(real_image, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_next_step, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_out, feature_d = self.discriminator.predict(real_image, verbose=0)

                # 3) generator training through combined (adversarial + feature)
                g_loss = self.combined.train_on_batch(
                    [image], [real_image, feature_d]
                )

                # 4) fact-matching loss
                fm_loss = 0.0
                for _ in range(fact_matching_cnt):
                    fm_loss += self.generator.train_on_batch(image, real_image)

                epoch_g_loss.append(g_loss[0])
                epoch_d_loss.append(d_loss)
                epoch_fm_loss.append(fm_loss)
                epoch_ff_loss.append(g_loss[1])

            g_loss_ary.append(np.mean(epoch_g_loss))
            d_loss_ary.append(np.mean(epoch_d_loss))
            fm_loss_ary.append(np.mean(epoch_fm_loss))
            ff_loss_ary.append(np.mean(epoch_ff_loss))

            if (ep + 1) % 10 == 0:
                print(
                    f"Epoch {ep+1} | "
                    f"D_loss={d_loss_ary[-1]:.4f} | "
                    f"G_loss={g_loss_ary[-1]:.4f} | "
                    f"FM={fm_loss_ary[-1]:.4f} | "
                    f"FF={ff_loss_ary[-1]:.4f}"
                )

        end_train_time = time.time()
        training_time = end_train_time - start_train_time
        print(f"[COREGAN] Total training time: {training_time:.1f} s")

        # Save loss curves
        epochs_arr = list(range(1, epochs + 1))
        loss_data = {
            "Epoch": epochs_arr,
            "D_Loss": d_loss_ary,
            "G_Loss": g_loss_ary,
            "FM_Loss": fm_loss_ary,
            "FF_Loss": ff_loss_ary,
        }
        loss_df = pd.DataFrame(loss_data)
        loss_df.to_csv(
            os.path.join(self.save_dir, "csv", "loss_final.csv"), index=False
        )

        # Quick plot
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot()
        ax1.plot(epochs_arr, g_loss_ary, "r", label="G loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("G-Loss", color="r")
        ax2 = ax1.twinx()
        ax2.plot(epochs_arr, d_loss_ary, "b", label="D loss")
        ax2.set_ylabel("D-Loss", color="b")
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "GD_loss.png"))
        plt.close(fig)

        # ---------------- TEST ----------------
        print("[COREGAN] Testing...")
        mse_list = []
        start_test_time = time.time()
        for i in range(x_test.shape[0]):
            track = x_test[i : i + 1]
            pred = self.generator.predict(track, verbose=0)  # (1, H, W, 1)

            mse = mean_squared_error(
                y_test[i].reshape(-1),
                pred[0].reshape(-1),
            )
            mse_list.append(mse)
        end_test_time = time.time()
        print(f"[COREGAN] Mean test MSE: {np.mean(mse_list):.6f}")
        print(f"[COREGAN] Total testing time: {end_test_time - start_test_time:.1f} s")

        np.savetxt(
            os.path.join(self.save_dir, "MSE_test.txt"),
            np.array(mse_list),
            fmt="%.9f",
        )
