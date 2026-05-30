from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

full_training_for_comparative_analysis = False 

if full_training_for_comparative_analysis: 
    ROOT = ROOT/"transformer_models"

# MODEL_NAME = "bert-base-uncased"

MODEL_NAME = "distilbert-base-uncased"

if MODEL_NAME == "distilbert-base-uncased":
    name = 'distilbert'
    pred_df_name = "distilbert_df.csv"
elif MODEL_NAME == "bert-base-uncased":
    name = 'bert'
    pred_df_name = "bert_df.csv"

OUTPUT_BASE = ROOT/"output"
NOTEBOOK = ROOT/"notebook"
LOG_BASE = ROOT/"logs"
RESULT_BASE = ROOT/"results"
DATA = ROOT/"data"


paths_to_create = [OUTPUT_BASE, NOTEBOOK, LOG_BASE, RESULT_BASE, DATA]

peft = True ## update to True is using peft_trained model

if peft: 
    paths_to_create.extend([
        LOG_BASE / "peft", 
        OUTPUT_BASE / "peft"/name, 
        DATA / "peft", 
        RESULT_BASE / "peft"/name  
    ])
else: 
    paths_to_create.extend([
        OUTPUT_BASE/name, 
        RESULT_BASE/name
    ])
    
for folder in paths_to_create:
    folder.mkdir(parents=True, exist_ok=True)

TUNE_DIR = f"./parameter_tuning/{name}"

LOG_FILE = f"{LOG_BASE}/peft/{name}_audit.log" if peft else f"{LOG_BASE}/{name}_audit.log"
LOG_DIR = f"{LOG_BASE}/peft/{name}.log" if peft else f"{LOG_BASE}/{name}.log"
OUTPUT_DIR = f"{OUTPUT_BASE}/peft/{name}" if peft else f"{OUTPUT_BASE}/{name}"
RESULT_DIR = f"{RESULT_BASE}/peft/{name}" if peft else f"{RESULT_BASE}/{name}"
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best_model"
PRED_DF_NAME = f"{DATA}/peft_{pred_df_name}" if peft else DATA/pred_df_name

CHECKPOINT_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth"
}


MODEL_TO_SERVE = OUTPUT_BASE/"peft"/"distilbert"/"best_model"


MAX_SEQ_LENGTH = 512
EPOCHS=5
BATCH_SIZE = 8
NUM_LABELS = 4
SEED = 200
BEST_MODEL_METRIC = "f1"


GRADIENT_ACC_STEPS=4
WEIGHT_DECAY=0.08978565486389789
LEARNING_RATE = 6.440037254992653e-05
WARMUP_STEP = 2

EARLY_STOP_THRESHOLD = 0.001
EARLY_STOP_PATIENCE = 3

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

