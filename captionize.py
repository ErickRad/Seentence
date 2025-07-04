import torch
from torchvision import transforms
from PIL import Image
import pickle
import argparse
import properties
from vocabBuilder import Vocabulary
from model import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loadVocab(path="data/vocab.pkl"):
    with open(path, "rb") as f:
        vocab = pickle.load(f)

    return vocab

def preprocessImage(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    
    return image

def main():
    parser = argparse.ArgumentParser(description="Image Captioning")
    parser.add_argument("--image-path", type=str, help="Path to input image")
    args = parser.parse_args()

    vocab = loadVocab()
    vocab_size = len(vocab)

    model = NeuralNetwork(properties.embedSize, properties.hiddenSize, vocab_size, device)
    model.load("checkpoints/encoder.pth", "checkpoints/decoder.pth")

    image = preprocessImage(args.image_path)
    caption = model.generateCaption(image, vocab)
    
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()