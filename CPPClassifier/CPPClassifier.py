import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import pandas as pd


class CPPClassifier:
    def __init__(self, models):
        self.models = models

    def predict(self, input_text):
        predictions = []

        for model_info in self.models:
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]

            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=70)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
                predictions.append(predicted_class)

        print(predictions)
        # The majority vote determines the final outcome.
        final_prediction = max(set(predictions), key=predictions.count)
        return final_prediction
    

models = [
    {
        "name": "ProtBert",
        "model_path": "../model/ProtBert_CPPSet1",
        "tokenizer_path": '../Rostlab/prot_bert'
    },
    {
        "name": "ProtBert_BFD",
        "model_path": "../model/ProtBert_BFD_CPPSet1/",
        "tokenizer_path": '../Rostlab/prot_bert_bfd'
    },
    {
        "name": "ProtElectra",
        "model_path": "../model/ProtElectra_CPPSet1/",
        "tokenizer_path": '../Rostlab/prot_electra_discriminator_bfd'
    }
]


loaded_models = []
for model_info in models:
    model = AutoModelForSequenceClassification.from_pretrained(model_info["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_path"])
    loaded_models.append({"model": model, "tokenizer": tokenizer})


cpp_classifier = CPPClassifier(loaded_models)


test_data = pd.read_csv('../dataset/CPPSet1-test.csv')  
sequences = test_data["Sequence"].tolist()
sequences = [' '.join(seq) for seq in sequences]
true_labels = test_data["Label"].tolist()


predicted_labels = []
for seq in sequences:
    pred = cpp_classifier.predict(seq)
    # print(pred)
    predicted_labels.append(pred)


accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
mcc = matthews_corrcoef(true_labels, predicted_labels)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC : {mcc:.4f}")
