# 🖼️ Seentence

**Seentence** is a deep learning project for automatic image captioning using PyTorch. It takes an image as input and generates a descriptive sentence about its contents, combining computer vision (CNN) and natural language processing (Transformer).

---

📚 **Project Structure**

  📂 datasets              → Custom PyTorch dataset logic
  
     →📝 dataset.py

  📂 models                → Encoder-Decoder architecture

     →🧠 captioningModel.py
    
     →🧩 cnnEncoder.py
    
     →📐 transformerDecoder.py
    

  📂 utils                 → Vocab, transforms, training tools
  
     →🔤 vocab.py
    
     →🎛️ dataAugmentation.py
    
     →🛠️ trainUtils.py
     

  📥 getData.py            → Download and unzip
  
  🏋️ train.py             → Train the model
  
  📊 evaluate.py          → Evaluate BLEU on test set
  
  🖼️ testImage.py         → Generate caption for a single image
  
  ⚙️ config.yaml           → Configs and hyperparameters
  
  📘 README.md             → You are here!

---

⚙️ **Requirements**

- Python 3.8+
- PyTorch
- torchvision
- pandas
- Pillow
- nltk
- pyyaml
- tqdm (opcional)

Install requirements:

pip install -r requirements.txt

---

📥 **Download the Dataset**

Run the script below to automatically download and extract the dataset:

python data/getData.py

It will:

- Download the image zip and annotations CSV from HuggingFace

- Extract images to data/images/

- Clean up the zip file

---

🔧 **Configuration**

Edit the config.yaml file to adjust:

-Paths (imagesPath, annotationsPath)

-Hyperparameters (batch size, learning rate, hidden size, etc.)

-Training settings (epochs, dropout, scheduler)

---

🚀 **Training**

To train the model:

python train.py

- This will save the best model to the path specified in config.yaml under     checkpoints folder.

---

📈 **Evaluation**

To evaluate your model on the test split (BLEU score):

python evaluate.py

---

🧪 **Test a Single Image**
You can run a custom image through the model to generate a caption (after training):

python testImage.py --image-path path/to/image.jpg

---

🧠 **Model Architecture**
CNN Encoder: Extracts visual features from input image

Transformer Decoder: Generates caption token-by-token

Vocabulary: Handles tokenization, mapping, padding

--

📝 **Dataset**
I have used the Flickr30k dataset, which contains 30,000 images and 5 human-written captions per image. Normalized to 224x224 and sent to model to recongnize edges, formats and shadows. After that, ponderate wheights to words to contruct a sentence about the image.

---

🤝  **Contributing**
Feel free to open issues, suggest improvements, or fork the repo.