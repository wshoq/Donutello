import os
import json
import random
from datasets import load_dataset, Features, Value, Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

# --- KONFIG ---
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
TRAIN_FILE = os.getenv("TRAIN_FILE", "data/train.jsonl")
VAL_FILE = os.getenv("VAL_FILE", "data/val.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
EPOCHS = int(os.getenv("EPOCHS", 3))
LR = float(os.getenv("LEARNING_RATE", 5e-5))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", 0.2))  # procent walidacji

# --- Sprawdzenie czy istnieje walidacja, jeśli nie -> podział ---
if not os.path.exists(VAL_FILE):
    print(f"[INFO] {VAL_FILE} nie znaleziono. Tworzę walidację {VAL_SPLIT*100:.0f}% z {TRAIN_FILE}...")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)
    split = int((1 - VAL_SPLIT) * len(lines))
    train, val = lines[:split], lines[split:]
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        f.writelines(train)
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        f.writelines(val)
    print(f"[INFO] Zapisano {len(train)} rekordów do {TRAIN_FILE} i {len(val)} do {VAL_FILE}")

# --- Wczytanie datasetu ---
features = Features({
    "input": Value("string"),
    "output": Value("string"),
    "image": Image()
})
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE}, features=features)

# --- Procesor Donut ---
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

def preprocess_function(example):
    pixel_values = processor(example["image"], return_tensors="pt").pixel_values.squeeze(0)
    labels = processor.tokenizer(
        example["output"],
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
        return_tensors="pt"
    )["input_ids"].squeeze(0)
    return {"pixel_values": pixel_values, "labels": labels}

dataset = dataset.map(preprocess_function, remove_columns=["image", "input", "output"])

# --- Argumenty trenera ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# --- Trener ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.tokenizer,
    data_collator=lambda data: {
        "pixel_values": torch.stack([f["pixel_values"] for f in data]),
        "labels": torch.stack([f["labels"] for f in data])
    }
)

# --- Trening ---
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Model zapisany w {OUTPUT_DIR}")
