import argparse

import torch
import yaml
from PIL import Image

from src.models.model_transforms import load_model


def get_labels() -> list[str]:
    """Reads in labels from labels.txt file.

    Returns:
        labels (list): list of labels.
    """
    with open("src/data/labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]


def read_params() -> dict:
    """
      Returns:
        params (dict): dictionary of parameters
    """
    with open("config/config.yml") as file:
        params = yaml.safe_load(file)["params"]
    return params


def predict(image: Image.Image, device: torch.device = "cpu") -> str:
    """
    Args:
        image_path (str): path to image
        device (str): device to run model on
    Returns:
        predicted_class (str): The predicted class

    """
    # wandb.login()
    # wandb.init(project="image_classification")
    model, transform = load_model()

    model.eval()

    transformed_image = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.inference_mode():
        output = model(transformed_image)

        predict_prob = torch.softmax(output, dim=1)

        predicted_label = torch.argmax(predict_prob, dim=1)

        predicted_class = get_labels()[predicted_label]
        # wandb.log({"image": wandb.Image(image, caption=predicted_class)})

        return predicted_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-IMAGE", "--IMAGE_PATH", type=str, help="path to image")
    args = parser.parse_args()
    predicted_label = predict(args.IMAGE_PATH)
    print("Predicted label:", predicted_label)
