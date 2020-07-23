# crop size of tiles from (500, 500)

TILE_SIZE = 128

# num of tiles to work with
SAMPLE_NUM = 10_000

# batch size
BATCH_SIZE = 10

# number of epochs
N_EPOCHS = 7

# log interval
LOG_INTERVAL = 100

# margin between anchor, pos and neg pairs in triples loss
MARGIN = 1

# embedding vector size
EMB_SIZE = 16

NUM_WORKERS = 8

CUDA = False

# loc2vec hyper-parameters
HP_ZOOM = [17]  # [17, 18]
HP_LR = [1e-4]  # [1e-3, 1e-4]
HP_WD = [0.1]
HP_STEP = [0.1]  # [0.1, 0.01]
