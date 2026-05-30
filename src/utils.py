import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from transformers import AutoModelForSequenceClassification 
from transformers import TrainingArguments
import evaluate 
from src import config 
import numpy as np 
import torch 

load_accuracy = evaluate.load("accuracy")
load_precision = evaluate.load("precision")
load_recall = evaluate.load("recall")
load_f1 = evaluate.load("f1")

use_gpu = torch.cuda.is_available()
use_bf16 = use_gpu and torch.cuda.is_bf16_supported()
use_fp16 = use_gpu and not use_bf16

def model(model_name:str, total_labels:int):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=total_labels, dtype="auto")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    
    preds, labels = eval_pred
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # If preds are still raw logits: shape = (n_samples, n_classes)
    # If preprocess_logits_for_metrics already ran: shape = (n_samples,)

    if preds.ndim > 1:
        preds = np.argmax(preds, axis=-1)

    preds = preds.reshape(-1)
    labels = labels.reshape(-1)


    accuracy = load_accuracy.compute(predictions=preds, references=labels)["accuracy"]
    precision = load_precision.compute(predictions=preds, references=labels, average="micro", zero_division=0)["precision"]
    recall = load_recall.compute(predictions=preds, references=labels, average='micro', zero_division=0)["recall"]
    f1 = load_f1.compute(predictions=preds, references=labels, average='micro')["f1"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_train_args(**kwargs):
    return TrainingArguments(
     output_dir = config.OUTPUT_DIR,
     num_train_epochs = config.EPOCHS,
     per_device_train_batch_size = config.BATCH_SIZE,
     gradient_accumulation_steps = config.GRADIENT_ACC_STEPS,
     gradient_checkpointing = True,
     weight_decay = config.WEIGHT_DECAY,
     learning_rate= config.LEARNING_RATE,
     warmup_steps = config.WARMUP_STEP, 
     eval_strategy = "epoch",
     eval_accumulation_steps = 10,
     eval_on_start = False,
     save_strategy = "epoch",
     load_best_model_at_end = True,
     logging_dir = config.LOG_DIR,
     metric_for_best_model=config.BEST_MODEL_METRIC,
     greater_is_better=True,
     max_grad_norm=1.0,
     dataloader_num_workers = 0, #gpu
     dataloader_persistent_workers=False,
     dataloader_prefetch_factor=None,
     restore_callback_states_from_checkpoint=True,
     neftune_noise_alpha=6.982938544247626,
     use_cache=False,
     bf16=use_bf16,
     fp16=use_fp16,
     dataloader_pin_memory=use_gpu,
     report_to = "tensorboard",
     **kwargs
    )