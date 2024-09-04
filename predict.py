import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json
from train import build_classifier  # Import if defined in train.py

def parse_args():
    parser = argparse.ArgumentParser(description='Predict image class.')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top K classes to return')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names mapping')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--model', type=str, choices=['vgg16', 'densenet121'], default='vgg16', help='Model architecture')
    return parser.parse_args()

def load_checkpoint(path='checkpoint.pth', model_name='vgg16'):
    checkpoint = torch.load(path, map_location='cpu')
    model_name = checkpoint['model_architecture']
    hidden_units = checkpoint['hidden_units']
    num_classes = checkpoint['num_classes']
    
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    # Build the classifier and assign it to the model
    model.classifier = build_classifier(model, hidden_units, num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, checkpoint['class_to_idx']


def process_image(image_pth):
    image = Image.open(image_pth)
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int((256 / width) * height)
    else:
        new_height = 256
        new_width = int((256 / height) * width)
    image = image.resize((new_width, new_height))
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image) / 255.0
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    np_image = (np_image - means) / stds
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def predict(image_path, model, class_to_idx, topk=5):
    processed_image = process_image(image_path)
    image_tensor = torch.from_numpy(processed_image).float().unsqueeze(0)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    ps = torch.exp(outputs)
    top_p, top_class = ps.topk(topk, dim=1)
    probs = top_p[0].cpu().numpy().tolist()
    classes_idx = top_class[0].cpu().numpy().tolist()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in classes_idx]
    return probs, classes

def display_prediction(image_path, model, class_to_idx, cat_to_name, topk=5):
    probs, classes = predict(image_path, model, class_to_idx, topk)
    class_names = [cat_to_name[str(class_idx)] for class_idx in classes]
    print("Top {} Predictions:".format(topk))
    for i in range(topk):
        print(f"{class_names[i]}: {probs[i]:.4f}")

def load_category_names(json_path):
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def main():
    args = parse_args()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model, class_to_idx = load_checkpoint(args.checkpoint, args.model)
    model.to(device)
    cat_to_name = load_category_names(args.category_names)
    display_prediction(args.image_path, model, class_to_idx, cat_to_name, args.top_k)

if __name__ == "__main__":
    main()
