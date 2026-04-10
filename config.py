import os

SEED = 42

# Default paths (can be overridden by CLI flags or env vars)
DATA_DIR = os.environ.get("BANANA_DATASET_ROOT", r"C:\path\to\your\dataset")
RESULTS_DIR = os.environ.get("BANANA_RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))

# Default 5-class unified task; set BANANA_NUM_CLASSES=9 for 9-class mode.
CLASS_NAMES_5 = ["nitrogen", "potassium", "magnesium", "iron", "zinc"]
CLASS_NAMES_9 = [
    "nitrogen",
    "potassium",
    "magnesium",
    "iron",
    "zinc",
    "healthy",
    "calcium",
    "manganese",
    "sulphur",
]

NUM_CLASSES = int(os.environ.get("BANANA_NUM_CLASSES", "5"))

BATCH_SIZE = 32
VAL_SPLIT = 0.2
IMG_SIZE_MOB = (224, 224)
IMG_SIZE_EFF = (224, 224)

EPOCHS_HEAD = 12
EPOCHS_FT1 = 12
EPOCHS_FT2 = 10

FOCAL_GAMMA = 2.0
CE_MIX = 0.30
EARLY_STOP_PATIENCE = 10


def active_classes():
    if NUM_CLASSES == 5:
        return CLASS_NAMES_5
    if NUM_CLASSES == 9:
        return CLASS_NAMES_9
    raise ValueError("BANANA_NUM_CLASSES must be 5 or 9")
