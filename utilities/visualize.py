from __future__ import annotations

from random import randint
import os

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import torch

import tensorflow as tf


class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=None, name=None, dtype=None):
        super(CustomMeanIoU, self).__init__(
            num_classes=num_classes, name=name, dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
        

COLORMAP = ([0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255])

class PredictionMasks:
    def __init__(
        self,
        trained_model,
        dataset_loader,
        inv_transform,
        number_of_classes,
    ):
        self.model = trained_model
        self.dataset_loader = dataset_loader
        self.inv_transform = inv_transform
        self.num_classes = number_of_classes
        self.predictor = trained_model

    def display_overlay_predictions_for_test_set(
        self,
        how_many_images: int,
        figure_size: tuple[int, int],
        colormap=COLORMAP,
        randomly: bool = True,
        export_to_file: bool = False,
    ):
        # if randomly:
        #     test_dataset = self.dataset.get_shuffled_test_dataset()
        # else:
        #     _, _, test_dataset = self.dataset.generate_datasets()

        i = 0
        should_break = False
        for images, y_true in self.dataset_loader:
            images = images.type(torch.FloatTensor).cuda()
            y_pred = self.predictor(images)
            _, y_pred = torch.max(y_pred, axis = 1)
            
            for image, mask_true, mask_pred in zip(images, y_true, y_pred):
                image = self.inv_transform(image.transpose(0,2).cpu().numpy()).transpose(0,2).numpy()
                mask_true = mask_true.cpu().numpy()
                mask_pred = mask_pred.detach().cpu().numpy()
                print(np.unique(mask_pred, return_counts=True))
                if i < how_many_images:
                    mask_true = self.decode_segmentation_mask(
                        mask_true, colormap, self.num_classes
                    )
                    mask_pred = self.decode_segmentation_mask(
                        mask_pred, colormap, self.num_classes
                    )
                    overlay = self.get_overlay(image, mask_pred)
                    overlay_original = self.get_overlay(image, mask_true)

                    # miou_score = 0
                    miou_score = self.get_miou_score_for_single_prediction(
                        mask_true.copy(), mask_pred.copy()
                    )
                    print(overlay_original.shape, mask_true.shape, mask_pred.shape)

                    self.plot_single_prediction(
                        [image, overlay_original, mask_true, overlay, mask_pred],
                        miou_score,
                        figure_size=figure_size,
                        should_save_to_file=export_to_file,
                    )
                    i += 1
                else:
                    should_break = True
                    break
            if should_break:
                break

    def get_overlay(self, image, colored_mask):
        overlay = np.asarray(Image.blend(Image.fromarray(image, 'RGB'), Image.fromarray(colored_mask, 'RGB'), 0.5))
        return overlay

    def get_miou_score_for_single_prediction(
        self, real_mask: np.array, predicted_mask: np.array
    ):
        miou = CustomMeanIoU(self.num_classes)
        miou.update_state(real_mask, predicted_mask)
        return miou.result().numpy()

    @staticmethod
    def decode_segmentation_mask(
        landcover_mask, custom_colormap: list[list[float]], num_classes: int
    ) -> np.array:
        """
        Transforms Landcover dataset's masks to RGB image.

        Args:
            landcover_mask: prediction or true mask;
            custom_colormap: user-defined colormap; len(custom_colormap) == num_classes;
                E.g. [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
                    [[R, G, B], [R, G, B], ...]
            num_classes: number of classes;
        """
        if len(custom_colormap) != num_classes:
            raise AttributeError("")
        r = np.zeros_like(landcover_mask).astype(np.uint8)
        g = np.zeros_like(landcover_mask).astype(np.uint8)
        b = np.zeros_like(landcover_mask).astype(np.uint8)
        for i in range(0, num_classes):
            idx = landcover_mask == i
            r[idx] = custom_colormap[i][0]
            g[idx] = custom_colormap[i][1]
            b[idx] = custom_colormap[i][2]
        rgb = np.stack([r, g, b], axis=2)

        return rgb

    @staticmethod
    def plot_single_prediction(
        images: list[np.array],
        miou_score: float,
        figure_size: tuple[int, int] = (10, 6),
        should_save_to_file: bool = False,
    ) -> None:
        score = round(miou_score * 100, 2)

        sub_names = [
            "Image",
            "Ground truth mask\n superimposed on the image",
            "Ground truth mask",
            "Predicted mask\n superimposed on the image",
            f"Predicted mask\nMean IoU = {score}%",
        ]

        fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=figure_size)

        for i, (name, image) in enumerate(zip(sub_names, images)):
            axes[i].set_title(name, size=16)
            axes[i].axis("off")
            if image.shape[-1] == 3:
                axes[i].imshow(image)
            else:
                axes[i].imshow(image)
        if should_save_to_file:
            dir_path = os.path.abspath(f"results/prediction_plots")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            filename = f"meanIoU_{score}_percent__{randint(1000, 9999)}"
            filepath = dir_path + f"/{filename}.png"
            plt.savefig(filepath)
        plt.show()