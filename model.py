import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from plot import LossPlotter
from train import log
from colorama import init, Fore
import properties

init(autoreset=True)

class PositionalEncoding(nn.Module):
    def __init__(self, dimModel, maxLen=5000):
        super().__init__()
        pe = torch.zeros(maxLen, dimModel)
        position = torch.arange(0, maxLen).unsqueeze(1).float()
        divTerm = torch.exp(torch.arange(0, dimModel, 2).float() * -(math.log(10000.0) / dimModel))

        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FeedForward(nn.Module):
    def __init__(self, dimModel, dimFF, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dimModel, dimFF)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dimFF, dimModel)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, dimModel, heads, dimFF, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dimModel, heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dimModel, dimFF, dropout)
        self.norm1 = nn.LayerNorm(dimModel)
        self.norm2 = nn.LayerNorm(dimModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, srcMask=None):
        attnOutput, _ = self.attn(x, x, x, attn_mask=srcMask)
        x = self.norm1(x + self.dropout(attnOutput))
        x = self.norm2(x + self.dropout(self.ff(x)))

        return x

class DecoderLayer(nn.Module):
    def __init__(self, dimModel, heads, dimFF, dropout=0.1):
        super().__init__()
        self.selfAttn = nn.MultiheadAttention(dimModel, heads, dropout=dropout, batch_first=True)
        self.crossAttn = nn.MultiheadAttention(dimModel, heads, dropout=dropout, batch_first=True)

        self.ff = FeedForward(dimModel, dimFF, dropout)

        self.norm1 = nn.LayerNorm(dimModel)
        self.norm2 = nn.LayerNorm(dimModel)
        self.norm3 = nn.LayerNorm(dimModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgtMask=None, memoryMask=None):
        selfAttn, _ = self.selfAttn(tgt, tgt, tgt, attn_mask=tgtMask)
        tgt = self.norm1(tgt + self.dropout(selfAttn))
        crossAttn, _ = self.crossAttn(tgt, memory, memory, key_padding_mask=memoryMask)
        tgt = self.norm2(tgt + self.dropout(crossAttn))
        tgt = self.norm3(tgt + self.dropout(self.ff(tgt)))

        return tgt

class VisionTransformerEncoder(nn.Module):
    def __init__(self, embedSize, imageSize=224, channels=3, numLayers=4, heads=8, dimFF=3072, dropout=0.1):
        super().__init__()
        self.patchSize = properties.patchSize
        self.numPatches = (imageSize // self.patchSize) ** 2

        patchDim = channels * self.patchSize * self.patchSize

        self.linear = nn.Linear(patchDim, embedSize)
        self.posEmbed = PositionalEncoding(embedSize, maxLen=self.numPatches)
        self.layers = nn.ModuleList([EncoderLayer(embedSize, heads, dimFF, dropout) for _ in range(numLayers)])

    def forward(self, images):
        B, C, H, W = images.size()
        p = self.patchSize

        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(B, -1, C * p * p)

        tokens = self.linear(patches)
        tokens = self.posEmbed(tokens)

        for layer in self.layers:
            tokens = layer(tokens)

        return tokens

class TransformerEncoder(nn.Module):
    def __init__(self, inputDim, dimModel, numLayers=4, heads=8, dimFF=3072, dropout=0.1, maxLen=5000):
        super().__init__()
        self.posEncoder = PositionalEncoding(dimModel, maxLen)
        self.layers = nn.ModuleList([
            EncoderLayer(dimModel, heads, dimFF, dropout) for _ in range(numLayers)
        ])

    def forward(self, x, srcMask=None):
        x = self.posEncoder(x)

        for layer in self.layers:
            x = layer(x, srcMask)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, tgtVocabSize, dimModel, numLayers=4, heads=8, dimFF=3072, dropout=0.1, maxLen=5000):
        super().__init__()
        self.embed = nn.Embedding(tgtVocabSize, dimModel)
        self.posEncoder = PositionalEncoding(dimModel, maxLen)

        self.layers = nn.ModuleList([
            DecoderLayer(dimModel, heads, dimFF, dropout) for _ in range(numLayers)
        ])

        self.fcOut = nn.Linear(dimModel, tgtVocabSize)

    def generateSquareSubsequentMask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt, memory, tgtMask=None, memoryMask=None):
        tgtEmb = self.embed(tgt)
        tgtEmb = self.posEncoder(tgtEmb)

        if tgtMask is None:
            tgtMask = self.generateSquareSubsequentMask(tgt.size(1)).to(tgt.device)

        for layer in self.layers:
            tgtEmb = layer(tgtEmb, memory, tgtMask, memoryMask)

        return self.fcOut(tgtEmb)

class NeuralNetwork:
    def __init__(self, embedSize, hiddenSize, vocabSize, device,
                 encoderLayers=4, decoderLayers=4, heads=8, dimFF=3072, dropout=0.1):
        self.device = device
        self.embedSize = embedSize
        self.vocabSize = vocabSize

        self.visionTransformer = VisionTransformerEncoder(embedSize).to(device)
        self.linearProj = nn.Identity()

        self.transformerEncoder = TransformerEncoder(
            embedSize, 
            embedSize, 
            encoderLayers, 
            heads, 
            dimFF, 
            dropout).to(device)
        
        self.transformerDecoder = TransformerDecoder(
            vocabSize, 
            embedSize, 
            decoderLayers, 
            heads, 
            dimFF, 
            dropout).to(device)

        self.scaler = GradScaler()

    def generateSquareSubsequentMask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(self.device)

    @staticmethod
    def cosineWithDelayScheduler(optimizer, warmup_steps, plateau_steps, total_steps, eta_min):
        def lrLambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            
            elif step < warmup_steps + plateau_steps:
                return 1.0
            
            else:
                progress = (step - warmup_steps - plateau_steps) / max(1, total_steps - warmup_steps - plateau_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))

                return cosine * (1 - eta_min / properties.learningRate) + eta_min / properties.learningRate
            
        return LambdaLR(optimizer, lrLambda)

    def train(self, trainLoader, criterion, optimizer, numEpochs, vocab, sampleImage=None):
        plotter = LossPlotter()

        self.visionTransformer.train()
        self.transformerEncoder.train()
        self.transformerDecoder.train()

        scheduler = NeuralNetwork.cosineWithDelayScheduler(
            optimizer,
            warmup_steps=properties.warmupSteps,
            plateau_steps=properties.plateauSteps,
            total_steps=len(trainLoader) * numEpochs,
            eta_min=properties.etaMin
        )

        for epoch in range(numEpochs):
            totalLoss = 0

            loop = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{numEpochs}",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [ {elapsed}/{remaining} ,{rate_fmt} {postfix} ]")

            for images, captions, _ in loop:
                images = images.to(self.device)
                captions = captions.to(self.device)

                optimizer.zero_grad()

                with autocast(device_type=self.device.type):
                    visionTokens = self.visionTransformer(images)
                    memory = self.transformerEncoder(self.linearProj(visionTokens))

                    tgtInput = captions[:, :-1]
                    tgtOutput = captions[:, 1:]

                    tgtMask = self.generateSquareSubsequentMask(tgtInput.size(1))
                    outputs = self.transformerDecoder(tgtInput, memory, tgtMask=tgtMask)

                    outputs = outputs.reshape(-1, outputs.size(-1))
                    tgtOutput = tgtOutput.contiguous().view(-1)

                    loss = criterion(outputs, tgtOutput)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                totalLoss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")

                if scheduler:
                    scheduler.step()

            plotter.add(epoch, loss.item())
            plotter.plot()

            avgLoss = totalLoss / len(trainLoader)
            log(f"\n[{Fore.GREEN}✓{Fore.RESET}] Average loss: {avgLoss:.4f}\n")
            
            if sampleImage is not None:
                exampleCaption = self.generateCaption(sampleImage, vocab)
                log(f"Sample caption: {exampleCaption}\n")
    
    @torch.no_grad()
    def generateCaption(self, image, vocab, maxLength=properties.maxLength, topK=properties.topK, temperature=properties.temperature):
        self.visionTransformer.eval()
        self.transformerEncoder.eval()
        self.transformerDecoder.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() != 4:
            raise ValueError(f"Image tensor must be 3D or 4D, got {image.dim()}D")

        image = image.to(self.device)

        with autocast(device_type=self.device.type):
            tokens = self.visionTransformer(image)
            memory = self.linearProj(tokens)
            memory = self.transformerEncoder(memory)

        captionIndices = []
        inputIds = torch.tensor([[vocab.stoi["<SOS>"]]], device=self.device)

        for _ in range(maxLength):
            tgtMask = self.generateSquareSubsequentMask(inputIds.size(1))

            with autocast(device_type=self.device.type):
                output = self.transformerDecoder(inputIds, memory, tgtMask=tgtMask)
                output = output[:, -1, :]
                output = output / temperature
                probs = torch.softmax(output, dim=-1)
                topk_vals, topk_indices = torch.topk(probs, k=topK)
                sampled = torch.multinomial(topk_vals.squeeze(0), 1)
                predictedId = topk_indices[0, sampled.item()].item()

            if predictedId == vocab.stoi.get("<EOS>", None):
                break

            if predictedId != vocab.stoi.get("<UNK>", -1):
                captionIndices.append(predictedId)

            inputIds = torch.cat([inputIds, torch.tensor([[predictedId]], device=self.device)], dim=1)

        captionWords = [vocab.itos[idx] for idx in captionIndices if idx in vocab.itos and vocab.itos[idx] != "<UNK>"]

        return " ".join(captionWords)
    
    def save(self, encoderPath, decoderPath):
        torch.save(self.visionTransformer.state_dict(), encoderPath)

        torch.save({
            'transformerEncoder': self.transformerEncoder.state_dict(),
            'transformerDecoder': self.transformerDecoder.state_dict(),
            'linearProj': self.linearProj.state_dict(),
        }, decoderPath)

    def load(self, encoderPath, decoderPath):
        self.visionTransformer.load_state_dict(torch.load(encoderPath, map_location=self.device))
        checkpoint = torch.load(decoderPath, map_location=self.device)
        
        self.transformerEncoder.load_state_dict(checkpoint['transformerEncoder'])
        self.transformerDecoder.load_state_dict(checkpoint['transformerDecoder'])
        self.linearProj.load_state_dict(checkpoint['linearProj'])