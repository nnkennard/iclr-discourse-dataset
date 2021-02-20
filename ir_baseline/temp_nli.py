from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset, load_metric

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "qnli"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16


actual_task = "mnli" if task == "mnli-mm" else task
metric = load_metric('glue', actual_task)
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]
def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
#num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def model_init(num_labels):
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


def hyp_search():
  trainer = Trainer(
      model_init=model_init,
      args=args,
      train_dataset=encoded_dataset["train"],
      eval_dataset=encoded_dataset[validation_key],
      tokenizer=tokenizer,
      compute_metrics=compute_metrics
  )

  best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

  print(best_run)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
def main():
  model = model_init(3)

  dataset = load_dataset("glue", actual_task)
  print(actual_task)
  print(dataset)
  encoded_dataset = dataset.map(preprocess_function, batched=True)
  validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
  trainer = Trainer(
      model,
      args,
      train_dataset=encoded_dataset["train"],
      eval_dataset=encoded_dataset[validation_key],
      tokenizer=tokenizer,
      compute_metrics=compute_metrics
  )

  trainer.train()

  trainer.evaluate()





if __name__ == "__main__":
  main()
