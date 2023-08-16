import argparse
from typing import List

import matplotlib.pyplot as plt
import torch
import torchvision
import yaml


def predict_and_plot_image(
        model_path: str,
        image_path: str,
        class_names: List[str] = None,
        transform=None,
        true_label: str = None,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model_path (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        true_label (str, optional): The true label of the image
        device (torch.device, optional): target device to compute on. Defaults to "cuda" else "cpu.

    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        predict_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Load in model from path and make sure it's on the right device
    model = torch.load(model_path, map_location=device)

    # 5. Make sure the model is on the target device
    model.to(device)

    # 6. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_prediction = model(target_image.to(device))

    # 7. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_predicted_probs = torch.softmax(target_image_prediction, dim=1)

    # 8. Convert prediction probabilities -> prediction labels
    target_image_predicted_label = torch.argmax(target_image_predicted_probs, dim=1)

    # 9. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )
    if class_names:
        title = (f"Prediction: {class_names[target_image_predicted_label.cpu()]} | "
                 f"Prob: {target_image_predicted_probs.max().cpu():.3f}")
    else:
        title = f"Prediction: {target_image_predicted_label} | Prob: {target_image_predicted_probs.max().cpu():.3f}"

    if true_label:
        if true_label == class_names[target_image_predicted_label.cpu()]:
            plt.title(f"{title}", c="green")
        else:
            plt.title(f"{title} True_Label: {true_label}", c="red")

    plt.axis(False)


def read_config():
    with open("config/config.yml") as file:
        config = yaml.safe_load(file)["params"]
        return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict and plot image')
    parser.add_argument('--model', type=str, help='model path', default=read_config()['model_path'])
    parser.add_argument('--image', type=str, help='image path')
    parser.add_argument('--class_names', type=str, help='class names', default=read_config()['class_names'])
    parser.add_argument('--true_label', type=str, help='true label')
    parser.add_argument('--device', type=str, help='device', default="cpu")

    args = parser.parse_args()

    predict_and_plot_image(model_path=args.model,
                           image_path=args.image,
                           class_names=args.class_names,
                           true_label=args.true_label,
                           device=args.device)

    plt.show()
