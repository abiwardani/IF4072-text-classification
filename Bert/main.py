from unicodedata import name
import pandas as pd
import numpy as np
import nltk
import collections
import torch
from transformers import BertTokenizer
from sklearn import preprocessing
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
nltk.download('punkt')

# running program using external model and do testing


def run(model_path, test_dataset_path, stopwords_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Copy the model to the GPU.
    model.to(device)

    # Reading test file
    test = pd.read_csv(test_dataset_path)
    test = data_cleansing(test, stopwords_path)
    run_test(test, model, tokenizer, device)


def run_test(test, model, tokenizer, device):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    test_sentences = test["text_cleansing"]
    test_labels = test["label"]

    # For every sentence...
    for sent in test_sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,                      # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=256,           # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,   # Construct attn. masks.
            return_tensors='pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(test_labels)

    # Set the batch size.
    batch_size = 32

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on test set

    print('Predicting labels for {:,} test sentences...'.format(
        len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')
    # Calculate accuracy for test dataset
    total_accuracy = 0
    for batch_num in range(len(predictions)):
        total_accuracy += flat_accuracy(
            predictions[batch_num], true_labels[batch_num])

    total_accuracy = total_accuracy/len(predictions)
    print('Accuracy on test dataset: {}'.format(total_accuracy))


def get_frequent_word(df):
    text = " ".join(list(df['text_a'].str.lower()))
    word_list = word_tokenize(text)
    word_count = dict(collections.Counter(word_list))
    d_word_freq = pd.DataFrame(
        data={'word': list(word_count.keys()), 'freq': list(word_count.values())})

    return d_word_freq


def cleansing(text, stopword=None):
    word_list = word_tokenize(text.lower())
    word_list = [word for word in word_list if len(word) > 2]
    word_list = [word for word in word_list if word.isalnum()]
    if stopword == None:
        text = ' '.join(word_list)
    else:
        word_list = [word for word in word_list if word not in stopword]
        text = ' '.join(word_list)

    return text


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def data_cleansing(test, stopwords_path):
    stopwords = list(pd.Series(pd.read_csv(stopwords_path)))

    for i in tqdm(range(len(test))):
        test.loc[i, 'text_cleansing'] = cleansing(
            test.loc[i, 'text_a'], stopword=stopwords)

    # dict mapping
    labels = ["no", "yes"]
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    test["label"] = le.transform(test["label"])

    return test


if name == "__main__":
    run("model", "datasets/test.csv", "datasets/stopwords.csv")
