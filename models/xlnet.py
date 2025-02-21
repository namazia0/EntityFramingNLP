import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class EntityFramingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        entity_mention = item['entity_mention']
        article_content = item['article_content']
        main_role = item['main_role']
        fine_grained_roles = item['fine_grained_roles']

        input_text = f"Entity: {entity_mention} \n Context: {article_content}"

        encoded = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'main_role': torch.tensor(main_role, dtype=torch.float),
            'fine_grained_roles': torch.tensor(fine_grained_roles, dtype=torch.float)
        }

class EntityFramingModel:
    def __init__(self, model_name, num_main_roles, num_fine_grained_roles, max_length, learning_rate):
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetForSequenceClassification.from_pretrained(
            model_name, num_labels=num_main_roles + num_fine_grained_roles
        )
        self.num_main_roles = num_main_roles
        self.num_fine_grained_roles = num_fine_grained_roles
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def train(self, train_dataloader, num_epochs, criterion_main, criterion_fine_grained):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                main_role_labels = batch['main_role'].to(self.device)
                fine_grained_labels = batch['fine_grained_roles'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.forward(input_ids, attention_mask)

                logits = outputs.logits
                main_role_logits = logits[:, :self.num_main_roles]
                fine_grained_logits = logits[:, self.num_main_roles:]

                main_role_loss = criterion_main(main_role_logits, main_role_labels)
                fine_grained_loss = criterion_fine_grained(fine_grained_logits, fine_grained_labels)
                loss = main_role_loss + fine_grained_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

    def predict(self, dataloader, threshold=0.5):
      self.model.eval()
      main_role_predictions, fine_grained_predictions, true_main_roles, true_fine_grained_roles = [], [], [], []

      with torch.no_grad():
          for batch in dataloader:
              input_ids = batch['input_ids'].to(self.device)
              attention_mask = batch['attention_mask'].to(self.device)
              main_role_labels = batch['main_role'].to(self.device)
              fine_grained_labels = batch['fine_grained_roles'].to(self.device)

              outputs = self.forward(input_ids, attention_mask)
              logits = outputs.logits

              main_role_logits = logits[:, :self.num_main_roles]
              fine_grained_logits = logits[:, self.num_main_roles:]

              main_role_predictions.append((torch.sigmoid(main_role_logits) > threshold).cpu().numpy())
              fine_grained_predictions.append((torch.sigmoid(fine_grained_logits) > threshold).cpu().numpy())

              true_main_roles.append(main_role_labels.cpu().numpy())
              true_fine_grained_roles.append(fine_grained_labels.cpu().numpy())

      return main_role_predictions, fine_grained_predictions, true_main_roles, true_fine_grained_roles

def prepare_data(data_file, main_roles, fine_grained_roles):
    data = []
    df = pd.read_csv(data_file)

    for _, row in df.iterrows():
        entity_mention = row['entity_mention']
        article_content = row['article_content']
        main_role = row['main_role']
        
        fine_grained_roles_1 = row['fine-grained_roles_1'].split("\t") if pd.notna(row['fine-grained_roles_1']) else []
        fine_grained_roles_2 = row['fine-grained_roles_2'].split("\t") if pd.notna(row['fine-grained_roles_2']) else []
        merged_fine_grained_roles = list(set(fine_grained_roles_1 + fine_grained_roles_2))
        
        main_role_label = [main_role] if pd.notna(main_role) else []
        
        data.append({
            'article_id': row['article_id'],
            'entity_mention': entity_mention,
            'start_offset': row['start_offset'],
            'end_offset': row['end_offset'],
            'article_content': article_content,
            'main_role': main_role_label,
            'fine_grained_roles': merged_fine_grained_roles
        })

    mlb_main = MultiLabelBinarizer(classes=main_roles)
    mlb_fine = MultiLabelBinarizer(classes=fine_grained_roles)

    all_main_roles = [item['main_role'] for item in data]
    all_fine_grained_roles = [item['fine_grained_roles'] for item in data]

    binary_main_roles = mlb_main.fit_transform(all_main_roles)
    binary_fine_grained_roles = mlb_fine.fit_transform(all_fine_grained_roles)

    for idx, item in enumerate(data):
        item['main_role'] = binary_main_roles[idx]
        item['fine_grained_roles'] = binary_fine_grained_roles[idx]

    return data, mlb_main, mlb_fine

def prepare_submission_file(data, main_role_predictions, fine_grained_predictions, mlb_main, mlb_fine, output_file):
    submission_data = []
    
    main_role_predictions = np.array(main_role_predictions)
    fine_grained_predictions = np.array(fine_grained_predictions)
    
    for idx, item in enumerate(data):
        article_id = item['article_id']
        entity_mention = item['entity_mention']
        start_offset = item['start_offset']
        end_offset = item['end_offset']

        predicted_main_role = mlb_main.inverse_transform(main_role_predictions[idx:idx + 1])[0]
        predicted_fine_grained_roles = mlb_fine.inverse_transform(fine_grained_predictions[idx:idx + 1])[0]

        main_role_str = predicted_main_role[0] if len(predicted_main_role) > 0 else "Innocent"
        fine_grained_roles_str = "\t".join(predicted_fine_grained_roles) if len(predicted_fine_grained_roles) > 0 else "None"

        submission_data.append({
            "article_id": article_id,
            "entity_mention": entity_mention,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "main_role": main_role_str,
            "fine_grained_roles": fine_grained_roles_str
        })

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_file, sep="\t", index=False)
    print(f"Submission file saved to {output_file}")

def main():
    model_name = "xlnet-base-cased"
    max_length = 512
    batch_size = 8
    num_epochs = 3
    learning_rate = 2e-5
    main_roles = ['Protagonist', 'Antagonist', 'Innocent']
    fine_grained_roles = ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous',
                      'Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot',
                      'Forgotten', 'Exploited', 'Victim', 'Scapegoat']

    data, mlb_main, mlb_fine = prepare_data("combined_training_data.csv", main_roles, fine_grained_roles)
    tokenizer = XLNetTokenizer.from_pretrained(model_name)

    # Split data into training and validation sets
    train_data = data[:int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)):]

    train_dataset = EntityFramingDataset(train_data, tokenizer, max_length)
    val_dataset = EntityFramingDataset(val_data, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights for imbalanced data
    binary_main_roles = np.array([item['main_role'] for item in data])
    binary_fine_grained_roles = np.array([item['fine_grained_roles'] for item in data])

    main_classes = np.arange(len(main_roles))
    fine_classes = np.arange(len(fine_grained_roles))

    main_class_weights = compute_class_weight('balanced', classes=main_classes, y=binary_main_roles.argmax(axis=1))
    fine_class_weights = compute_class_weight('balanced', classes=fine_classes, y=binary_fine_grained_roles.argmax(axis=1))

    main_class_weights = torch.tensor(main_class_weights, dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    fine_class_weights = torch.tensor(fine_class_weights, dtype=torch.float).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
  
    criterion_main = torch.nn.BCEWithLogitsLoss(pos_weight=main_class_weights)
    criterion_fine_grained = torch.nn.BCEWithLogitsLoss(pos_weight=fine_class_weights)

    # Initialize model
    model = EntityFramingModel(
        model_name, 
        num_main_roles=len(main_roles), 
        num_fine_grained_roles=len(fine_grained_roles), 
        max_length=max_length, 
        learning_rate=learning_rate
    )

    # Training
    model.train(train_dataloader, num_epochs, criterion_main, criterion_fine_grained)

    # Evaluating
    main_role_predictions, fine_grained_predictions, true_main_roles, true_fine_grained_roles = model.predict(val_dataloader)

    binary_main_predictions = (np.vstack(main_role_predictions) > 0.5).astype(int)
    binary_fine_predictions = (np.vstack(fine_grained_predictions) > 0.5).astype(int)

    prepare_submission_file(val_data, binary_main_predictions, binary_fine_predictions, mlb_main, mlb_fine, "submission_file.tsv")

    binary_true_main_roles = np.vstack(true_main_roles).astype(int)
    binary_true_fine_grained_roles = np.vstack(true_fine_grained_roles).astype(int)

    main_role_f1 = f1_score(binary_true_main_roles, binary_main_predictions, average="micro")
    main_role_exact_match = accuracy_score(binary_true_main_roles, binary_main_predictions)

    fine_grained_f1 = f1_score(binary_true_fine_grained_roles, binary_fine_predictions, average="micro")
    fine_grained_exact_match = accuracy_score(binary_true_fine_grained_roles, binary_fine_predictions)

    print(f"Main Roles - Micro-averaged F1-score: {main_role_f1:.4f}, Exact Match Ratio: {main_role_exact_match:.4f}")
    print(f"Fine Grained Roles - Micro-averaged F1-score: {fine_grained_f1:.4f}, Exact Match Ratio: {fine_grained_exact_match:.4f}")
    print("Main Roles - Classification Report:")
    print(classification_report(binary_true_main_roles, binary_main_predictions, target_names=mlb_main.classes_))
    print("Fine Grained Roles - Classification Report:")
    print(classification_report(binary_true_fine_grained_roles, binary_fine_predictions, target_names=mlb_fine.classes_))


if __name__ == "__main__":
    main()
