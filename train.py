import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('data_dir', type=str, help='Directory with the training data')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='File path to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def build_classifier(model, hidden_units, num_classes):
    if isinstance(model, models.VGG):
        return nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )
    elif isinstance(model, models.DenseNet):
        return nn.Sequential(
            nn.Linear(model.classifier.in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )

def main():
    args = parse_args()
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Directories
    train_dir = f"{args.data_dir}/train"
    valid_dir = f"{args.data_dir}/valid"
    test_dir = f"{args.data_dir}/test"
    
    # Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(valid_dir, transform=data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=64, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    }

    # Load category names
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    num_classes = len(image_datasets['train'].classes)

    # Model selection
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = 25088
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_features = model.classifier.in_features

    for param in model.parameters():
        param.requires_grad = False

    classifier = build_classifier(model, args.hidden_units, num_classes)
    model.classifier = classifier

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    model.to(device)

    # Training
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects = (outputs.argmax(dim=1) == labels).float().sum().item()
            running_corrects += corrects
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                corrects = (outputs.argmax(dim=1) == labels).float().sum().item()
                val_corrects += corrects
                val_total_samples += inputs.size(0)

        val_loss /= val_total_samples
        val_acc = val_corrects / val_total_samples
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Save checkpoint
    checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'model_architecture': args.arch,
        'hidden_units': args.hidden_units,
        'num_classes': num_classes,
        'loss': criterion
    }
    torch.save(checkpoint, args.save_dir)
    print(f"Checkpoint saved to {args.save_dir}")



if __name__ == "__main__":
    main()
