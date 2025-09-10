import os
import json
import random
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image
import torch

# --- KONFIG ---
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
DATA_DIR = os.getenv("DATA_DIR", "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
EPOCHS = int(os.getenv("EPOCHS", 3))
LR = float(os.getenv("LEARNING_RATE", 5e-5))

# --- Walidacja datasetu ---
if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"[ERROR] Nie znaleziono {TRAIN_FILE}")

if not os.path.exists(VAL_FILE):
    print(f"[INFO] {VAL_FILE} nie znaleziono. Tworzę walidację 20% z {TRAIN_FILE}...")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)
    split_idx = int(len(lines) * 0.8)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        f.writelines(val_lines)
    print(f"[INFO] Zapisano {len(train_lines)} rekordów do {TRAIN_FILE} i {len(val_lines)} do {VAL_FILE}")

# --- Wczytanie datasetu ---
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

# --- Procesor i model Donut ---
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# --- Preprocessing ---
def preprocess_function(example):
    image_path = os.path.join(DATA_DIR, example["image_path"])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Nie znaleziono obrazu {image_path}")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze(0)
    labels = processor.tokenizer(
        example["output"],
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
        return_tensors="pt"
    )["input_ids"].squeeze(0)
    return {"pixel_values": pixel_values, "labels": labels}

dataset = dataset.map(preprocess_function, remove_columns=["image_path", "input", "output"])

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
