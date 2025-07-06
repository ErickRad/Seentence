epochs              = 100
learningRate.       = 1e-4
etaMin              = 1e-5
warmupSteps         = 12000
plateauSteps        = 36000
patchSize           = 16

threshold           = 1
maxLength           = 30
maxSamples          = None
topK                = 30
temperature         = 1.1

batchSize           = 64
embedSize           = 256
hiddenSize          = 512

IMAGE_DIR_TRAIN     = "data/train2017"
IMAGE_DIR_VAL.      = "data/val2017"
CAPTION_FILE_TRAIN  = "data/annotations/captions_train2017.json"
CAPTION_FILE_VAL.   = "data/annotations/captions_val2017.json"
VOCAB_PATH          = 'data/vocab.pkl'
LOGS_PATH           = 'logs/'