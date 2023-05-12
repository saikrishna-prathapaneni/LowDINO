from colossalai.amp import AMP_TYPE

# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 32
NUM_EPOCHS = 200
WARMUP_EPOCHS = 40

CONFIG = dict(fp16=dict(mode=AMP_TYPE.TORCH))