import os
import json
import random
from datasets import Dataset, DatasetDict
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image
import torch
from glob import glob

# --- KONFIG ---
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
DATA_DIR = os.getenv("DATA_DIR", "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
EPOCHS = int(os.getenv("EPOCHS", 3))
LR = float(os.getenv("LEARNING_RATE", 5e-5))
MAX_LENGTH = 512

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
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # znajdź wszystkie strony dla zlecenia
            base_name = item["image_path"].rsplit("-page_", 1)[0]
            pages = sorted(glob(os.path.join(DATA_DIR, f"{base_name}-page_*.png")))
            if not pages:
                pages = [os.path.join(DATA_DIR, item["image_path"])]
            item["image_paths"] = pages
            data.append(item)
    return data

train_data = load_jsonl(TRAIN_FILE)
val_data = load_jsonl(VAL_FILE)
dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data)
})

# --- Procesor i model ---
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# --- Preprocessing ---
def preprocess_function(example):
    # wczytanie wszystkich stron i połączenie w listę tensorów
    images = []
    for img_path in example["image_paths"]:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[ERROR] Nie znaleziono obrazu {img_path}")
        image = Image.open(img_path).convert("RGB")
        images.append(processor(image, return_tensors="pt").pixel_values.squeeze(0))
    # łączymy obrazy w tensor (lista -> batch)
    pixel_values = torch.stack(images)  # [num_pages, 3, H, W]

    labels = processor.tokenizer(
        example["output"],
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )["input_ids"].squeeze(0)

    return {"pixel_values": pixel_values, "labels": labels}

dataset = dataset.map(preprocess_function, remove_columns=["image_path", "input", "output", "image_paths"])

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
