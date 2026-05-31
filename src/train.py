import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Trainer, EarlyStoppingCallback, DataCollatorWithPadding, AutoTokenizer
from pathlib import Path
import time
from src import config
import re
import json
from src.seed import set_seed
from src.load_data import prepare_dataset
from src.log_builder import build_logger
from src import utils

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

def tokenization(example):
    return tokenizer(example["text"],
                     truncation=True,
                     max_length=config.MAX_SEQ_LENGTH,
                    )

def is_complete_checkpoint(path: Path) -> bool:
    existing = {p.name for p in path.iterdir() if p.is_file()}
    model_file_exists = "model.safetensors" in existing or "pytorch_model.bin" in existing or "adapter_model.safetensors" in existing
    return model_file_exists and config.CHECKPOINT_FILES.issubset(existing)

def find_latest_checkpoint(dir):
    directory = Path(dir)
    if not directory.exists():
        return None

    checkpoints = []
    for p in directory.iterdir():
        if p.is_dir() and re.match(r"checkpoint-\d+$", p.name):
            step = int(p.name.split('-')[-1])
            if is_complete_checkpoint(p):
                checkpoints.append((step, p))
            else:
                print(f"Skipping incomplete checkpoint: {p}")

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    print(f"Resuming from checkpoint: {checkpoints[-1][1]}")
    return str(checkpoints[-1][1])

def main(config):
    set_seed()
    global tokenizer

    logger, file_handler = build_logger()
    logger.removeHandler(file_handler)
    logger.addHandler(file_handler)

    logger.debug("Loading train and validation datasets for model training")

    train, validation = prepare_dataset()

    data_collator = DataCollatorWithPadding(tokenizer)

    train_data = train.map(tokenization, batched=True)
    val_data = validation.map(tokenization, batched=True)
    
    if config.TRAIN_SUBSET_SIZE and config.EVAL_SUBSET_SIZE:
        train_data = train_data.select(range(config.TRAIN_SUBSET_SIZE))
        eval_data = eval_data.select(range(config.EVAL_SUBSET_SIZE))

    logger.debug("Building model and training arguments")

    model = utils.model(config.MODEL_NAME, config.NUM_LABELS)

    training_args = utils.build_train_args()
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset= train_data.select(range(config.TRAIN_SUBSET_SIZE)),
        eval_dataset = val_data.select(range(config.EVAL_SUBSET_SIZE)),
        processing_class = tokenizer,
        data_collator = data_collator,
        compute_metrics = utils.compute_metrics,
        preprocess_logits_for_metrics=utils.preprocess_logits_for_metrics, #keep eye
        callbacks = [EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOP_PATIENCE,
                                          early_stopping_threshold= config.EARLY_STOP_THRESHOLD),]

    )

    logger.debug(f"Model selected as >>> {config.MODEL_NAME} <<< ")
    logger.info("============BATCH TRAINING INFORMATION============ "
        f"Train size: {len(trainer.train_dataset)} "
        f"Batch size: {config.BATCH_SIZE} "
        f"Epochs: {config.EPOCHS} "
        f"Steps per epoch: {len(trainer.train_dataset) // config.BATCH_SIZE} "
        f"Approx total steps: {(len(trainer.eval_dataset) // config.BATCH_SIZE) * config.EPOCHS}"
    )

    last_checkpoint = find_latest_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        logger.info (f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        logger.info("No checkpoint found. Starting fresh training run.")

    start = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.evaluate()
    stop = time.time()
    train_time = {"Training_time": round(stop-start, 3)}

    with open(f"{config.RESULT_DIR}/train_time.json", 'w') as f:
        json.dump(train_time, f, indent=2)

    best_model_path = f"{training_args.output_dir}/best_model"
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    logger.info("Model training completed successfully with best model and tokenizer saved")
    logger.removeHandler(file_handler)

if __name__ == "__main__":
    main(config)
    
    