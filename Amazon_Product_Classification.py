
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ============================
# Load Data
# ============================

df = pd.read_csv("amazon_products_sales_data_cleaned.csv")

# ============================
# Basic Exploration (EDA)
# ============================

print(df.head())
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.describe())

print(df["purchased_last_month"].head())
print(df[["product_title", "product_category", "sustainability_tags"]].head())
print(df["sustainability_tags"].value_counts())

# ============================
# Label Creation
# ============================

def demand_label(value):
    if value > 975:
        return "High"
    elif 485 < value <= 975:
        return "Medium"
    else:
        return "Low"

df["demand_label"] = df["purchased_last_month"].apply(demand_label)

print(df[["purchased_last_month", "demand_label"]].head(15))
print(df["demand_label"].value_counts())

label_map = {"Low": 0, "Medium": 1, "High": 2}
df["label"] = df["demand_label"].map(label_map)

print(df[["demand_label", "label"]].head(10))

# ============================
# Text Preparation
# ============================

df["product_title"] = df["product_title"].fillna("")
df["product_category"] = df["product_category"].fillna("")
df["sustainability_tags"] = df["sustainability_tags"].fillna("")

df["text_input"] = (
    df["product_title"] + " " +
    df["product_category"] + " " +
    df["sustainability_tags"]
)

print(df["text_input"].head(3))
print(df[["text_input", "demand_label"]].sample(5))

# ============================
# Train Test Split
# ============================

X = df["text_input"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Train label counts:")
print(y_train.value_counts())
print("Test label counts:")
print(y_test.value_counts())

# ============================
# Tokenization
# ============================

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(
    list(X_train),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

test_encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

train_labels = torch.tensor(list(y_train))
test_labels = torch.tensor(list(y_test))

# ============================
# Model Setup
# ============================

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ============================
# Dataset Class
# ============================

class AmazonDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = AmazonDataset(train_encodings, y_train.tolist())
eval_dataset = AmazonDataset(test_encodings, y_test.tolist())

# ============================
# Training Arguments
# ============================

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# ============================
# Train Model
# ============================

trainer.train()

# ============================
# Evaluation
# ============================

eval_results = trainer.evaluate(eval_dataset)
print(eval_results)

predictions = trainer.predict(eval_dataset)
logits = predictions.predictions
predicted_labels = logits.argmax(axis=1)

y_true = test_labels.numpy()
y_pred = predicted_labels

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["Low", "Medium", "High"]))

# ============================
# Inference Example
# ============================

product_title = "Wireless Bluetooth Headphones"
product_category = "Electronics"
sustainability_tags = "Recyclable"

text_input = product_title + " " + product_category + " " + sustainability_tags

inputs = tokenizer(
    text_input,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

inputs = {key: val.to(device) for key, val in inputs.items()}

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(axis=1).item()

reverse_label_map = {0: "Low", 1: "Medium", 2: "High"}
predicted_label = reverse_label_map[predicted_class_id]
print("Predicted demand:", predicted_label)

# ============================
# Save Model
# ============================

os.makedirs("model", exist_ok=True)
model.save_pretrained("model")
tokenizer.save_pretrained("model")
