"""
Model definition and inference logic for the DistilBERT news classifier.
Must match the training notebook's setup exactly (same label order, same
max sequence length) or the downloaded weights won't map correctly.
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Must match the label2id mapping used during training, sorted alphabetically
# from the training notebook's `categories = sorted(train_df['category'].unique())`
CATEGORIES = ['business', 'entertainment', 'politics', 'sport', 'tech']
LABEL2ID = {cat: i for i, cat in enumerate(CATEGORIES)}
ID2LABEL = {i: cat for cat, i in LABEL2ID.items()}

MAX_LENGTH = 512


def load_model_and_tokenizer(weights_path, device):
    """
    Loads the fine-tuned DistilBERT model and its tokenizer. Raises clearly
    if the checkpoint doesn't match the expected architecture, rather than
    failing silently or producing garbage predictions.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer


def predict(model, tokenizer, text, device):
    """
    Runs inference on a single piece of text. Returns the predicted category
    and the full probability distribution across all five categories, not
    just the winning label, since the confidence spread is informative on
    its own (a 95% vs 35% top prediction means very different things).
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().tolist()

    probabilities = {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)}
    predicted_category = max(probabilities, key=probabilities.get)

    return predicted_category, probabilities
