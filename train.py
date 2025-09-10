import os
import json
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

# --- KONFIG ---
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
TRAIN_FILE = os.getenv("TRAIN_FILE", "data/train.jsonl")
VAL_FILE = os.getenv("VAL_FILE", "data/val.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
EPOCHS = int(os.getenv("EPOCHS", 3))
LR = float(os.getenv("LEARNING_RATE", 5e-5))

# --- Jeśli brak val.jsonl → dzielimy train.jsonl ---
if not os.path.exists(VAL_FILE):
    print(f"[INFO] {VAL_FILE} nie znaleziono. Tworzę walidację 20% z {TRAIN_FILE}...")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[INFO] Zapisano {len(train_data)} rekordów do {TRAIN_FILE} i {len(val_data)} do {VAL_FILE}")

# --- Wczytanie datasetu ---
dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

# --- Procesor Donut ---
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# --- Funkcja preprocess ---
def preprocess_function(example):
    # otwieramy obraz z image_path
    image_path = example["image_path"]
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

# --- Mapowanie datasetu ---
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
