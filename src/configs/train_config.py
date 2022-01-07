
# Train settings
RESIZE = 128
BATCH_SIZE = 8
MAX_VIDEO_FRAMES = 60
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
EPOCHS = 0
ES_PATIENCE = 5
ES_DELTA = 0

# Data, saving, etc settings
DATASET_FOLDER = '/mnt/d/MTX_hackathon/backuppp/data/training/multi_instance/final'
SAVE_MODEL = True
SAVE_MODEL_PTH = '/mnt/d/MTX_hackathon/best__.pt'
LOAD_MODEL = True
LOAD_MODEL_PTH = '/mnt/d/MTX_hackathon/best.pt'

# Others
EVALUATE_ON_TEST_SET = True

#! TODO: change paths to repo folder structure