
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer

with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=101)

effnetb2.load_state_dict(torch.load(f="Pretrained_EffNetB2_Feature_Extractor_Food101.pth",
                                    map_location=torch.device("cpu")))

def predict(img) -> tuple[dict, float]:
    start_time = timer()
    img = effnetb2_transforms(img).unsqueeze(0)  # unsqueeze = add batch dimension on 0th dimension
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    pred_time = round(timer() - start_time, 4)

    return pred_labels_and_probs, pred_time


example_list = [["examples/" + example] for example in os.listdir("examples")]

title = "Food Vision 101 🍕👁️💪"
description = "An [EffNetB2 feature extractor](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html) computer vision model to classify 101 classes of food from the [Food101](https://pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html) dataset."
article = "Create at [KeivanJamali.com](http://keivanjamali.com)."

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"),
                             gr.Number(label="Prediction Time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

demo.launch()

