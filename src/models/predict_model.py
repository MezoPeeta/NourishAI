import torch
from PIL import Image
from model_transforms import create_effnetb2_model
import argparse
import yaml

def class_names():
    with open("data/labels.txt", "r") as f:
        class_names = f.read().splitlines()
    return class_names


def read_params():
    """
    Args:
        config_path (str): path to config file
    Returns:
        params (dict): dictionary of parameters
    """
    with open("config/config.yml") as file:
        params = yaml.safe_load(file)["params"]
    return params

def predict(image_path: str, device: torch.device = "cpu") -> str:
    """
    Args:
        model_path (str): path to model
        image_path (str): path to image
        device (str): device to run model on
    Returns:
        predicted_class (str): The predicted class

    """

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

        predicted_class = class_names()[predicted_label]

        return predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    args = parser.parse_args()
    print(predict(args.image_path))    
