from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.Toxicity_classifier.data_clean import male_comments , female_comments

model_name = "unitary/unbiased-toxic-roberta"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model.to(device)
#### male comments setup
df_eval = male_comments.copy().head(2000)
male_texts = df_eval['comment_text'].astype(str).tolist()
batch_size = 32
all_male_probs = []

### female comments setup
female_comments_copy = female_comments.copy().head(2000)
female_texts = female_comments_copy['comment_text'].astype(str).tolist()
all_female_probs = []

with torch.no_grad():
    for i in range(0, len(male_texts), batch_size):
        batch = male_texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)


        toxicity_idx = model.config.label2id["toxicity"]
        probs = torch.sigmoid(outputs.logits)[:, toxicity_idx]
        print(f'the male probable toxicity is : {probs}')
        all_male_probs.append(probs.cpu())


    for i in range(0,len(female_texts),batch_size):
        batch = female_texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = 256,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        toxicity_idx = model.config.label2id["toxicity"]
        probs = torch.sigmoid(outputs.logits)[:, toxicity_idx]
        print(f'the female probable toxicity is : {probs}')
        all_female_probs.append(probs.cpu())



### male toxicity to pandas/numpy
all_probs = torch.cat(all_male_probs)
print("N =", all_probs.numel())
print("mean toxicity =", all_probs.mean().item())
print("median toxicity =", all_probs.median().item())

male_to_numpy = all_probs.numpy()
df_eval['toxicity_predicted'] = male_to_numpy

print(df_eval.columns)
print(df_eval)


#### female toxicity to pandas/numpy
all_probs = torch.cat(all_female_probs)
print("N =", all_probs.numel())
print("mean toxicity =", all_probs.mean().item())
print("median toxicity =", all_probs.median().item())

female_to_numpy = all_probs.numpy()
female_comments_copy['toxicity_predicted'] = female_to_numpy

print(female_comments_copy.columns)
print(female_comments_copy)


