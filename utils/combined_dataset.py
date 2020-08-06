import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as func
from numpy import ndarray
from PIL import Image
from torchvision.datasets import CIFAR10, VisionDataset


class CombinedDataset(VisionDataset):
    def __init__(self, root=None, transform=None):
        super(CombinedDataset, self).__init__(root, transform=transform)

        if root is None:
            root: str = os.path.join(os.getcwd(), "data", "processed", "cifar-10-combined", "train.pt")

        self.data: ndarray
        self.targets: List[int]
        self.data, self.targets = self.generate_combined_data(root)

    def __getitem__(self, index: int) -> Tuple[ndarray, int]:
        """__getitem__ method.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img: ndarray = self.data[index]
        target: int = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _downscale_and_combine(self, img_list: List[Image.Image]) -> ndarray:
        """Downscale 4 PIL Images, convert to ndarray, and combine in a grid.

        Each image must be square. The downscaling halves the resolution.

        Args:
            img_list (List[Image.Image]): List of 4 square PIL Images.

        Returns:
            numpy.ndarray: Combined image, the same resolution as the original 4
            images. Dtype is uint8.
        """
        assert len(img_list) == 4  # Assert batch size = 4.
        assert all(img.height == img.width for img in img_list)  # Assert square images.

        downscaled: List[ndarray] = [np.asarray(func.resize(img, int(img.height // 2))) for img in img_list]

        hstack_top: ndarray = np.concatenate((downscaled[0], downscaled[1]), 1)
        hstack_bottom: ndarray = np.concatenate((downscaled[2], downscaled[3]), 1)
        return np.concatenate((hstack_top, hstack_bottom), 0)

    def generate_combined_data(self, root_path):
        data_exists: bool = os.path.isfile(root_path)

        if data_exists:
            combined_data: Tuple[ndarray, List[int]] = torch.load(root_path, "train.pt")
        else:
            dataset = CIFAR10(root=os.path.join(os.getcwd(), "data", "raw"), train=True, download=True)

            indexes: List[List[int]] = [[idx for idx, class_idx in enumerate(dataset.targets)
                                        if class_idx == k]
                                        for k in range(10)]
            new_data: ndarray = np.zeros((len(dataset) // 4, 32, 32, 3), dtype=np.uint8)
            new_targets: List[int] = [0] * (len(dataset) // 4)

            for class_idx in range(len(dataset.classes)):
                new_count: int = 0
                to_combine: List = [0] * 4
                for img_num, img_idx in enumerate(indexes[class_idx]):
                    to_combine[img_num % 4] = dataset[img_idx][0]

                    if (img_num + 1) % 4 == 0:
                        new_idx: int = (class_idx * len(indexes[class_idx]) // 4) + new_count
                        new_data[new_idx] = self._downscale_and_combine(to_combine)
                        new_targets[new_idx] = class_idx
                        new_count += 1

            combined_data: Tuple[ndarray, List[int]] = (new_data, new_targets)

            os.makedirs(os.path.dirname(root_path), exist_ok=True)
            torch.save(combined_data, root_path)

        return combined_data


if __name__ == "__main__":
    CombinedDataset()
