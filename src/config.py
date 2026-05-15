from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "bert-base-uncased"

if MODEL_NAME == "distilbert-base-uncased":
    name = 'distilbert'
    pred_df_name = "distilbert_df.csv"
elif MODEL_NAME == "bert-base-uncased":
    name = 'bert'
    pred_df_name = "bert_df.csv"

OUTPUT_DIR = ROOT/"output"/name
NOTEBOOK = ROOT/"notebook"
LOG_BASE = ROOT/"logs"
RESULT_DIR = ROOT/"results"/name
DATA = ROOT/"data"


for folder in [OUTPUT_DIR, NOTEBOOK, LOG_BASE, RESULT_DIR, DATA]:
    folder.mkdir(parents=True, exist_ok=True)

LOG_DIR = LOG_BASE/f"{name}.log"
TUNE_DIR = f"./parameter_tuning/{name}"

MAX_SEQ_LENGTH = 512
SEED = 200
EPOCHS=5
BATCH_SIZE = 16


GRADIENT_ACC_STEPS=4
WEIGHT_DECAY=0.08978565486389789
LEARNING_RATE = 6.440037254992653e-05
WARMUP_STEP = 2
BEST_MODEL_METRIC = "f1"


CHECKPOINT_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth"
}

NUM_LABELS = 4

EARLY_STOP_THRESHOLD = 0.001
EARLY_STOP_PATIENCE = 3
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

peft = True ## update to True is using peft_trained model

LOG_FILE = f"{LOG_BASE}/peft/{name}_audit.log" if peft else f"{LOG_BASE}/{name}_audit.log"
LOG_DIR = f"{LOG_BASE}/peft/{name}.log" if peft else f"{LOG_BASE}/{name}.log"
BEST_MODEL_PATH = OUTPUT_DIR/"peft"/"best_model" if peft else  OUTPUT_DIR/"best_model"
PRED_DF_NAME = f"{DATA}/peft_{pred_df_name}" if peft else DATA/pred_df_name