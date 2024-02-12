import os
import argparse
import random
import shutil
import torch
import logging

import numpy as np

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from model import resnet18

from xai import GradCAM, SHAP


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

argument_parser = argparse.ArgumentParser(
    description="This script explains the model predictions."
)
argument_parser.add_argument("-o", "--output_dir", required=True, type=str)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def explain_model(output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = resnet18()
    model.load_state_dict(torch.load(os.path.join("experiment", "last_epoch_model.pt")))
    model = model.eval()

    valid_transforms = [transforms.Resize(size=(224, 224)), transforms.ToTensor()]

    valid_dataset = datasets.ImageFolder(
        root="./data/valid",
        transform=valid_transforms,
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=1
    )

    for input_index, (inputs, labels) in enumerate(valid_loader):

        sample_filepath, _ = valid_loader.dataset.samples[input_index]
        sample_filename = os.path.splitext(os.path.basename(sample_filepath))[0]

        filedir = os.path.join(output_dir, "explanations", sample_filename)

        true_label = labels[0].item()

        shutil.copy2(
            sample_filepath, os.path.join(filedir, os.path.basename(sample_filepath))
        )

        shap_explainer = SHAP(model)
        shap_explainer.explain(
            inputs[0].numpy(),
            target=true_label,
            output_filepath=os.path.join(
                filedir, f"{sample_filename}_shap_{input_index}.png"
            ),
        )

        gradcam_explainer = GradCAM(model)
        gradcam_explainer.explain(
            inputs,
            target=true_label,
            output_filepath=os.path.join(
                filedir, f"{sample_filename}_gradcam_{input_index}.png"
            ),
        )


if __name__ == "__main__":

    args = argument_parser.parse_args()

    output_dir = args.output_dir

    explain_model(output_dir=output_dir)
