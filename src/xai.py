import cv2
import logging

import numpy as np

import torch
import torch.nn.functional as F

from captum.attr import KernelShap
from captum.attr import visualization as viz

import skimage.segmentation as segs

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GradCAM:
    """
    Based on this implementation: https://github.com/vickyliin/gradcam_plus_plus-pytorch
    """

    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.target_layer = dict([*self.model.named_modules()])[target_layer_name]
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, input_tensor, class_idx=None, retain_graph=False):
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        if class_idx is None:
            class_idx = outputs.argmax(dim=1)
        score = outputs[:, class_idx].squeeze()

        score.backward(retain_graph=retain_graph)

        gradients = self.gradients
        activations = self.activations
        b, k, _, _ = gradients.size()

        alpha = gradients.view(b, k, -1).mean(dim=2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(dim=1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(
            saliency_map,
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        saliency_map = (saliency_map - saliency_map.min()) / (
            saliency_map.max() - saliency_map.min() + 1e-8
        )

        return saliency_map, outputs

    def __call__(self, input_tensor, class_idx=None, retain_graph=False):
        return self.forward(input_tensor, class_idx, retain_graph)

    def explain(self, inputs, target, output_filepath, retain_graph=False):
        mask, _ = self.forward(inputs, target, retain_graph)
        self.visualize(inputs, mask[0][0].numpy(), output_filepath)

    @staticmethod
    def visualize_cam(mask, img):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
        b, g, r = heatmap.split(1)
        heatmap = torch.cat([r, g, b])

        result = heatmap + img.cpu()
        result = result.div(result.max()).squeeze()

        return heatmap, result


class SHAP:
    def __init__(self, model):
        self.model = model
        self.explainer = KernelShap(model)

    def explain(self, inputs, target, output_filepath):
        feature_mask = torch.from_numpy(
            segs.slic(
                np.moveaxis(inputs, 0, 2),
                n_segments=50,
                compactness=30,
                sigma=3,
            )
        )

        attributions = self.explainer.attribute(
            inputs=inputs, feature_mask=feature_mask, target=target
        )

        self.visualize(attributions, inputs, output_filepath)

        return attributions

    @staticmethod
    def visualize(attributions, img, output_filepath):

        if torch.max(attributions) == 0:
            logging.error("Could not produce shap heatmaps.")
            return

        plot, _ = viz.visualize_image_attr(
            np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            method="blended_heat_map",
            sign="all",
            outlier_perc=1,
            use_pyplot=False,
        )

        plot.savefig(output_filepath, bbox_inches="tight", pad_inches=0)
