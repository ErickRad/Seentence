import torch
import os
import pickle
import properties
import torch.nn as nn
import torch.optim as optim
from colorama import init, Fore
from datetime import datetime
from dataset import *
from model import *
from vocabBuilder import Vocabulary, loadCaptions

init(autoreset=True)
logPath = None
device = None

def log(message, end="\n"):
    global logPath

    if logPath is None:
        timestamp = datetime.now().strftime("%d.%m-%Hh%Mm")
        baseDir = properties.LOGS_PATH
        os.makedirs(baseDir, exist_ok=True)

        path = os.path.join(baseDir, f"{timestamp}.txt")
        i = 1
        while os.path.exists(path):
            path = os.path.join(baseDir, f"{timestamp}.{i}.txt")
            i += 1

        logPath = path

    cleanMessage = message.replace(Fore.GREEN, "").replace(Fore.RED, "").replace(Fore.RESET, "")
    
    print(message, end=end)

    with open(logPath, "a", encoding="utf-8") as logFile:
        logFile.write(cleanMessage + end)


def main():
    if os.path.exists(properties.VOCAB_PATH):
        log(f"[{Fore.GREEN}✓{Fore.RESET}] Looking for vocabulary... {properties.VOCAB_PATH}.")
        with open(properties.VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)
    else:
        os.makedirs(os.path.dirname(properties.VOCAB_PATH), exist_ok=True)

        log(f"[+] Loading captions...", end="\r")
        captions = loadCaptions(properties.CAPTION_FILE_TRAIN)
        log(f"[{Fore.GREEN}✓{Fore.RESET}] Loading captions... done.")

        log(f"[+] Building vocabulary...", end="\r")
        vocab = Vocabulary(threshold=properties.threshold)
        vocab.buildVocab(captions)
        log(f"[{Fore.GREEN}✓{Fore.RESET}] Building vocabulary... {len(vocab)} words on {properties.VOCAB_PATH}.")

        with open(properties.VOCAB_PATH, 'wb') as f:
            pickle.dump(vocab, f)

    log(f"[+] Looking for CUDA...", end="\r")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        log(f"[{Fore.GREEN}✓{Fore.RESET}] Looking for CUDA... {torch.version.cuda}.")
    else:
        device = torch.device("cpu")
        log(f"[{Fore.RED}X{Fore.RESET}] Looking for CUDA... not available. Resuming on CPU 💀😨.")

    model = NeuralNetwork(properties.embedSize, properties.hiddenSize, len(vocab), device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])

    optimizer = optim.Adam(
        list(model.visionTransformer.parameters()) +
        list(model.transformerEncoder.parameters()) +
        list(model.transformerDecoder.parameters()) +
        list(model.linearProj.parameters()),
        lr=properties.learningRate
    )

    log(f"[+] Starting training...\n")
    model.train(loader(vocab), criterion, optimizer, properties.epochs, vocab)
    log(f"[{Fore.GREEN}✓{Fore.RESET}] Starting training... done.")

    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/encoder.pth", "checkpoints/decoder.pth")

if __name__ == "__main__":
    main()
