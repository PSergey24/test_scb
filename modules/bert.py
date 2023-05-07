import os
import torch
import re
import string
import numpy as np
import pandas as pd
from torch import nn
from transformers import BertTokenizer
from torch.utils.data import Dataset
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymorphy2 import MorphAnalyzer


morph = MorphAnalyzer()
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
labels = {'Зарплатные проекты': 0,
          'Эквайринг': 1,
          'Открытие банковского счета': 2,
          'Бизнес-карта': 3
          }


class Dataset(Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text,
                               padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.7):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()

        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(768, 4),
        #     nn.ReLU()
        # )

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        # final_layer = self.classifier(pooled_output)

        return final_layer

    def save(self, way_to_model):
        folders = way_to_model.split('/')
        model_folder_path, file_name = '/'.join(folders[:-1]), folders[-1]

        if os.path.exists(model_folder_path + '/.DS_Store') is True:
            os.remove(model_folder_path + '/.DS_Store')

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        content = os.listdir(model_folder_path)
        if len(content) > 0:
            old_model_loss = float(content[0].replace('bert_epoch_', '').replace('.pth', '').split('_')[1])
            new_model_loss = float(file_name.replace('bert_epoch_', '').replace('.pth', '').split('_')[1])

            if new_model_loss < old_model_loss:
                file_name = os.path.join(model_folder_path, file_name)
                torch.save(self.state_dict(), file_name)
                os.remove(model_folder_path + '/' + content[0])
        else:
            file_name = os.path.join(model_folder_path, file_name)
            torch.save(self.state_dict(), file_name)


def main():
    np.random.seed(112)
    df = get_data()
    df = data_preprocessing(df)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])

    # df_train = df_train[:3200]

    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-6

    train(model, df_train, df_val, LR, EPOCHS)
    evaluate(model, df_test)


def get_data():
    df = pd.read_csv('data/train.csv', sep=';', encoding='utf-8')
    return df


def data_preprocessing(df):
    stop_words = set(stopwords.words('russian'))
    stop_words = add_stop_words(stop_words)

    df = filter_simple_phrases(df)
    df = filter_short_phrases(df)

    df = df.drop_duplicates(subset=['text_employer'])

    string.punctuation = string.punctuation.replace('-', '')
    df['text_employer'] = df['text_employer'].apply(lambda t: t.lower())
    df['text_employer'] = df['text_employer'].apply(replace_symbol)
    df['text_employer'] = df['text_employer'].apply(lambda x: remove_stopwords(x, stop_words))
    df['text_employer'] = df['text_employer'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
    df['text_employer'] = df['text_employer'].apply(lambda x: re.sub('\w*\d\w*', '', x))
    df['text_employer'] = df['text_employer'].apply(lemmatize)

    df = df[['text_employer', 'ACTION_ITEM_RESULT_PRODUCT_NAME']]
    df = df.rename(columns={'text_employer': 'text', 'ACTION_ITEM_RESULT_PRODUCT_NAME': 'category'})
    return df


def add_stop_words(stop_words):
    # seems that for classification task need to add stop words like 'hello'
    new_words = ['э', 'а', 'м', 'же', 'у', 'box', 'санкт-петербург', 'москва', 'это', 'лопух', 'вот', 'там', 'психоз', 'ну',
    'этот', 'уже', 'сантиметров', 'жесткая', 'она', 'прям', 'вообще', 'пошел', 'звоним', 'штоб', 'бога', 'на', 'максим',
    'или', 'мастер', 'кого', 'ой', 'по', 'выходной', 'давай', 'истории', 'пари', 'виктория']
    return set(list(stop_words) + new_words)


def filter_simple_phrases(df):
    return df[(df['text_employer'] != 'добрый_день') & (df['text_employer'] != 'здравствуйте')
              & (df['text_employer'] != 'добрый_день. добрый_день')
              & (df['text_employer'] != 'хорошо')
              & (df['text_employer'] != 'добрый')
              & (df['text_employer'] != 'а')]


def filter_short_phrases(df):
    return df[(df['text_employer'].str.len() > 100)]


def replace_symbol(x):
    return x.replace('_', ' ')


def remove_stopwords(doc, stop_words):
    filtered_doc = [token for token in word_tokenize(doc) if not token in stop_words]
    return " ".join(filtered_doc)


def lemmatize(doc):
    tokens = [morph.normal_forms(token)[0] for token in word_tokenize(doc)]
    return " ".join(tokens)


def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

        way_to_model = f'data/models/bert_epoch_{epoch_num}_{total_loss_val / len(val_data): .3f}.pth'
        model.save(way_to_model)


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


