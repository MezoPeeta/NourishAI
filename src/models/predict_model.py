import argparse

import torch
import wandb
import yaml
from PIL import Image

from model_transforms import create_effnetb2_model


def get_labels() -> list[str]:
    """Reads in labels from labels.txt file.

    Returns:
        labels (list): list of labels.
    """
    with open("../../data/labels.txt", "r") as f:
        labels = f.read().splitlines()
    return labels


def read_params() -> dict:
    """
      Returns:
        params (dict): dictionary of parameters
    """
    with open("../../config/config.yml") as file:
        params = yaml.safe_load(file)["params"]
    return params


def predict(image_path: str, device: torch.device = "cpu") -> str:
    """
    Args:
        image_path (str): path to image
        device (str): device to run model on
    Returns:
        predicted_class (str): The predicted class

    """
    wandb.login()
    wandb.init(project="image_classification")
    model, transform = create_effnetb2_model()
    model.load_state_dict(torch.load(read_params()["model_path"], map_location=device)["model_state_dict"])

    model.eval()

    image = Image.open(image_path)
    transformed_image = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.inference_mode():
        output = model(transformed_image)

        predict_prob = torch.softmax(output, dim=1)

        predicted_label = torch.argmax(predict_prob, dim=1)

        predicted_class = get_labels()[predicted_label]
        wandb.log({"image": wandb.Image(image, caption=predicted_class)})

        return predicted_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    args = parser.parse_args()
    predicted = predict(args.image_path)


