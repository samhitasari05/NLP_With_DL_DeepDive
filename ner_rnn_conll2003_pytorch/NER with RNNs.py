import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import zipfile
import time



# Add path to conlleval.py (assumed in ./conll2003)
sys.path.append("/scratch/dberhan4/conll2003")
from conlleval import evaluate_conll_file  

"""
The above imports provided by the client (in this case Prof. Lioa) the code has the evaluation metrics
accuracy, precision, F1 score

"""
# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =================================== Data Preprocessing Functions ============================
def lower_case(words):
    pattern = r'([A-Z][a-z]+)'
    def lower_match(match):
        return match.group(0).lower()
    return re.sub(pattern, lower_match, words)

def load_sentences(filename, max_sentences=None):
    sentences = []
    current_sentence = []
    max_len = 0
    sentence_count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if max_sentences and sentence_count >= max_sentences:
                break
            if line.startswith("-DOCSTART-") or not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    max_len = max(max_len, len(current_sentence))
                    current_sentence = []
                    sentence_count += 1
            else:
                tokens = line.split()
                word = lower_case(tokens[0])
                tag = tokens[-1]
                current_sentence.append([word, tag])
    if current_sentence:
        sentences.append(current_sentence)
        max_len = max(max_len, len(current_sentence))
    return sentences, max_len

def create_vocab(sentences):
    word_vocab = {'<pad>': 0, '<unk>': 1}
    tag_vocab = {'<pad>': 0, 'O': 1, 'B-ORG': 2, 'B-PER': 3, 'B-LOC': 4, 'B-MISC': 5,
                 'I-ORG': 6, 'I-PER': 7, 'I-LOC': 8, 'I-MISC': 9}
    word_idx = 2
    for sentence in sentences:
        for word, _ in sentence:
            if word not in word_vocab:
                word_vocab[word] = word_idx
                word_idx += 1
    return word_vocab, tag_vocab

def zero_padding(sentences, max_len):
    padded_sentences = []
    for sentence in sentences:
        padded_sentence = sentence.copy()
        while len(padded_sentence) < max_len:
            padded_sentence.append(['<pad>', '<pad>'])
        padded_sentences.append(padded_sentence)
    return padded_sentences

def get_index_mapping(sentences, word_vocab, tag_vocab):
    data, labels = [], []
    unk_idx = word_vocab['<unk>']
    oov_words = 0
    for sentence in sentences:
        sentence_indices, label_indices = [], []
        for word, tag in sentence:
            word_idx = word_vocab.get(word, unk_idx)
            if word not in word_vocab:
                oov_words += 1
            tag_idx = tag_vocab[tag]
            sentence_indices.append(word_idx)
            label_indices.append(tag_idx)
        data.append(sentence_indices)
        labels.append(label_indices)
    print(f"Found {oov_words} OOV words, mapped to <unk>")
    return np.array(data), np.array(labels)

def get_embeddings(word_vocab):
    embedding_dim = 300
    vocab_size = len(word_vocab)
    embeddings = np.zeros((vocab_size, embedding_dim))
    try:
        print("Loading Word2Vec model...")
        word2vec_model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        found_count = 0
        for word, idx in word_vocab.items():
            if word in word2vec_model:
                embeddings[idx] = word2vec_model[word]
                found_count += 1
            else:
                embeddings[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)
        print(f"Found {found_count}/{vocab_size} words in embeddings")
    except Exception as e:
        print(f"Error loading embeddings: {e}. Using random initialization.")
        embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    return embeddings

# =========================================== Dataset and Dataloader ==========================
class NERDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.LongTensor(data)
        self.labels = torch.LongTensor(labels)
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    def __len__(self):
        return len(self.data)

def create_dataloaders(train_data, train_labels, val_data, val_labels, test_data, test_labels, batch_size):
    train_dataset = NERDataset(train_data, train_labels)
    val_dataset = NERDataset(val_data, val_labels)
    test_dataset = NERDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader

# === Model Definitions ===
class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, pretrained_embeddings=None, bidirectional=False, update_embeddings=True):
        super(VanillaRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = update_embeddings
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        output = self.fc(rnn_out)
        return output

class LSTMNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, pretrained_embeddings=None, bidirectional=False, update_embeddings=True):
        super(LSTMNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = update_embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

class GRUNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, pretrained_embeddings=None, bidirectional=False, update_embeddings=True):
        super(GRUNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = update_embeddings
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        output = self.fc(gru_out)
        return output

# === Training and Evaluation Functions ===
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, model_name, patience=10):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model = None
    early_stop_counter = 0
    start_time = time.time()
    
    for epoch in tqdm(range(num_epochs), desc=f"Training {model_name}"):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, output.shape[-1]), target.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output.view(-1, output.shape[-1]), target.view(-1)).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 100 == 0:  # Print less frequently due to high epoch count
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f"Convergence Plot for {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"convergence_{model_name}.png")
    plt.close()
    
    print(f"Finished {model_name} in {time.time() - start_time:.2f}s")
    return model

def predict(model, data_loader, id_to_tag):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            predictions = output.argmax(dim=2)
            for i in range(predictions.size(0)):
                sentence_preds = [id_to_tag[pred.item()] for pred in predictions[i]]
                all_predictions.append(sentence_preds)
    return all_predictions

def save_predictions(sentences, predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent, preds in zip(sentences, predictions):
            for (word, gold), pred in zip(sent, preds):
                # Skip padding tokens
                if word == '<pad>' or gold == '<pad>':
                    continue
                # Ensure gold and pred tags are valid
                gold_tag = gold if gold in ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC'] else 'O'
                pred_tag = pred if pred in ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC'] else 'O'
                f.write(f"{word} {gold_tag} {pred_tag}\n")
            f.write("\n")  # Empty line between sentences

# === Main Execution ===
def main():
    # File paths
    train_file = "/scratch/dberhan4/conll2003/train.txt"
    valid_file = "/scratch/dberhan4/conll2003/valid.txt"
    test_file = "/scratch/dberhan4/conll2003/test.txt"
    
    # Load and preprocess data (use all files by removing max_sentences)
    print("Loading and preprocessing data...")
    train_sentences, train_max_len = load_sentences(train_file)  # Full dataset
    valid_sentences, valid_max_len = load_sentences(valid_file)  # Full dataset
    test_sentences, test_max_len = load_sentences(test_file)     # Full dataset
    max_len = max(train_max_len, valid_max_len, test_max_len)
    print(f"Maximum sentence length: {max_len}")
    
    # Create vocabulary
    print("Creating vocabulary...")
    word_vocab, tag_vocab = create_vocab(train_sentences)
    print(f"Vocabulary size: {len(word_vocab)}")
    id_to_tag = {v: k for k, v in tag_vocab.items()}
    
    # Pad sentences
    train_padded = zero_padding(train_sentences, max_len)
    valid_padded = zero_padding(valid_sentences, max_len)
    test_padded = zero_padding(test_sentences, max_len)
    
    # Convert to indices
    train_data, train_labels = get_index_mapping(train_padded, word_vocab, tag_vocab)
    valid_data, valid_labels = get_index_mapping(valid_padded, word_vocab, tag_vocab)
    test_data, test_labels = get_index_mapping(test_padded, word_vocab, tag_vocab)
    
    # Batch size
    batch_size = 16  # Fixed batch size
    print(f"Batch size: {batch_size}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, train_labels, valid_data, valid_labels, test_data, test_labels, batch_size)
    
    # Load embeddings
    embeddings = get_embeddings(word_vocab)
    
    # Model parameters
    vocab_size = len(word_vocab)
    embedding_dim = 300
    hidden_size = 256
    output_size = len(tag_vocab)
    learning_rate = 0.0001
    num_epochs = 5000  # Assignment requirement
    

    # Define all seven models
    models = [
        ("rnn_256_softmax", VanillaRNN(vocab_size, embedding_dim, hidden_size, output_size, embeddings, bidirectional=False)),
        ("birnn_256_softmax", VanillaRNN(vocab_size, embedding_dim, hidden_size, output_size, embeddings, bidirectional=True)),
        ("lstm_256_softmax", LSTMNER(vocab_size, embedding_dim, hidden_size, output_size, embeddings, bidirectional=False)),
        ("bilstm_256_softmax", LSTMNER(vocab_size, embedding_dim, hidden_size, output_size, embeddings, bidirectional=True)),
        ("gru_256_softmax", GRUNER(vocab_size, embedding_dim, hidden_size, output_size, embeddings, bidirectional=False)),
        ("bigru_256_softmax", GRUNER(vocab_size, embedding_dim, hidden_size, output_size, embeddings, bidirectional=True)),
        ("bigru_256_softmax_update_emb", GRUNER(vocab_size, embedding_dim, hidden_size, output_size, embeddings, bidirectional=True, update_embeddings=True))
    ]
    
    # Train and evaluate all models with pauses so that if hopper disconnects we don't have to restart from the beginning
    results = {}
    f1_scores = {}
    for i, (model_name, model) in enumerate(models):
        print(f"\nTraining {model_name} ({i+1}/{len(models)})...")
        model = model.to(device)
        trained_model = train_model(model, train_loader, val_loader, num_epochs, learning_rate, model_name, patience=10)
        torch.save(trained_model.state_dict(), f"model_{model_name}.pt")
        
        # Predict on test set
        predictions = predict(trained_model, test_loader, id_to_tag)
        output_file = f"output_{model_name}.txt"
        save_predictions(test_padded, predictions, output_file)

        try:
            # Process the output file to create properly formatted sequences for evaluation
            true_seqs, pred_seqs = [], []
            current_true_seq, current_pred_seq = [], []
            
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        # End of a sentence - add the collected sequences
                        if current_true_seq and current_pred_seq:
                            true_seqs.append(current_true_seq)
                            pred_seqs.append(current_pred_seq)
                        current_true_seq, current_pred_seq = [], []
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 3:
                        word, true_tag, pred_tag = parts[:3]
                        # Skip padding tokens
                        if word == '<pad>' or true_tag == '<pad>':
                            continue
                        # Make sure tags are valid for conlleval
                        if true_tag not in ['O', '<pad>'] and '-' not in true_tag:
                            true_tag = 'O'  # Default to O if not in expected format
                        if pred_tag not in ['O', '<pad>'] and '-' not in pred_tag:
                            pred_tag = 'O'  # Default to O if not in expected format
                        
                        current_true_seq.append(true_tag)
                        current_pred_seq.append(pred_tag)
            
            # Add the last sentence if not added already
            if current_true_seq and current_pred_seq:
                true_seqs.append(current_true_seq)
                pred_seqs.append(current_pred_seq)
            
            # Flatten sequences for evaluation if needed
            flat_true_seqs = [tag for seq in true_seqs for tag in seq]
            flat_pred_seqs = [tag for seq in pred_seqs for tag in seq]
            
            # Check for valid data
            if not flat_true_seqs or not flat_pred_seqs:
                raise ValueError("No valid sequences extracted from the output file")
            
            print(f"Evaluating {len(flat_true_seqs)} tags from {len(true_seqs)} sentences...")
            
            # Print a few examples for debugging
            print("Sample tags (true, pred):")
            for i in range(min(10, len(flat_true_seqs))):
                print(f"  {flat_true_seqs[i]}, {flat_pred_seqs[i]}")
            
            # Now evaluate using the conlleval function
            from conlleval import evaluate
            precision, recall, f1 = evaluate(flat_true_seqs, flat_pred_seqs, verbose=True)
            
            results[model_name] = (precision, recall, f1)
            f1_scores[model_name] = f1
            print(f"{model_name} - Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            traceback.print_exc()
            results[model_name] = (0, 0, 0)
            f1_scores[model_name] = 0

        # Pause after each model (except the last one)
        if i < len(models) - 1:
            input("Press Enter to continue to the next model...")


if __name__ == "__main__":
    main()