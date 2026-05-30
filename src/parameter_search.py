import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import utils
from src import config
import json
from src.load_data import prepare_dataset
from src.train import tokenizer, tokenization
from src.seed import set_seed
from src.log_builder import build_logger
from transformers import AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding


def model_init(trial):
    global config
    return AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=4)

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 2, 4),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 2),
        "neftune_noise_alpha": trial.suggest_float("neftune_noise_alpha", 0.1, 10.0)
    }


def compute_objective(metrics):
    print(metrics)
    return metrics["eval_f1"]

def main():

    set_seed()

    train_data, eval_data = prepare_dataset()

    data_collator = DataCollatorWithPadding(tokenizer)
    train_data = train_data.map(tokenization, batched=True)
    eval_data = eval_data.map(tokenization, batched=True)

    training_args = utils.build_train_args()
    training_args.output_dir = config.TUNE_DIR
    training_args.logging_dir = config.TUNE_DIR

    trainer = Trainer(
        model_init = model_init,
        args = training_args,
        train_dataset = train_data.select(range(5000)),
        eval_dataset = eval_data.select(range(1000)),
        processing_class = tokenizer,
        data_collator = data_collator,
        compute_metrics=utils.compute_metrics,
    )

    best_run = trainer.hyperparameter_search(
            hp_space = hp_space,
            compute_objective = compute_objective,
            n_trials=10,
            direction="maximize",
            backend="optuna",
        )

    tune_result = {}
    tune_result["objective"] = best_run.objective
    tune_result["hyperparameters"] = best_run.hyperparameters

    with open(f"{config.RESULT_DIR}/best_hyperparameters.json", "w") as p:
        json.dump(tune_result, p, indent=2)

    print("Hyperparameter search complete")

if __name__ == "__main__":
    main()
    
