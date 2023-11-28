#!/usr/bin/env python3
from transformers import Wav2Vec2Model, BertConfig, AutoTokenizer, BertForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from transformers.utils import PaddingStrategy
import evaluate
import numpy as np
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

class DataColloatorForTokenClassificationEmbeddings(DataCollatorForTokenClassification):
    def torch_call(self, features):
        import torch

       
        # Get the max sequence length   
        max_length = max(len(feature['inputs_embeds']) for feature in features)     

        # Initialize padded embeddings, attention masks, and labels
        batch_embeddings = []
        batch_attention_masks = []
        batch_labels = []

        for feature in features:
            embeddings = np.asarray(feature['inputs_embeds'])
            labels = feature['labels']

            # Calculate the padding length
            padding_length = max_length - len(embeddings)

            # Pad embeddings with zero vectors and create attention masks
            if padding_length > 0:
                padded_embeddings = torch.cat((torch.tensor(embeddings), torch.zeros((padding_length, embeddings.shape[1]))), axis=0)  
                attention_mask = [1]*len(embeddings) + [0]*padding_length
            else:
                padded_embeddings = torch.tensor(embeddings)
                attention_mask = [1]*len(embeddings)
            # Append to the batch lists
            batch_embeddings.append(padded_embeddings)
            batch_attention_masks.append(attention_mask)
            if labels is not None:

                # If the label is probability distribution, (2-D tensor) then we need to pad the labels with 0s
                if len(labels.shape) == 2:
                    padding_length = max_length - len(labels)
                    if padding_length > 0:
                        labels = torch.cat((labels, torch.zeros((padding_length, labels.shape[1]))), axis=0)
                    else:
                        labels = torch.tensor(labels)
                    batch_labels.append(labels)  
                    
                elif len(labels.shape) == 1:
                    # Then pad the labels
                    batch_labels.append(
                        torch.cat(
                            (labels, torch.tensor([-100]*padding_length, dtype=torch.int64)), 
                            dim=0,
                        )
                    )
                else:
                    raise ValueError(f"The shape of labels is not correct: {labels.shape}")

        # Convert lists to tensors
        batch_embeddings = torch.stack(batch_embeddings)
        batch_attention_masks = torch.tensor(np.array(batch_attention_masks), dtype=torch.int64)
        if len(batch_labels) > 0:
            batch_labels = torch.stack(batch_labels)
        else:
            batch_labels = None

        # Print the names and shapes
        batch = {
            "inputs_embeds": batch_embeddings[:,:512],
            "attention_mask": batch_attention_masks[:, :512],
            "labels": batch_labels[:, :512],
        }

        return batch

class TokenClassificationEmbeddingsDataset(Dataset):
    def __init__(self, data_path, split, downsample_rate = 3, label_smoothing = False):
        self.sr = downsample_rate
        self.kernel_size = 2
        self.label_smoothing = label_smoothing
        self.kernel = torch.exp(-torch.arange(-self.kernel_size, self.kernel_size+1)**2/2)
        with open(f"/work/dcml0714/uasr_rl/rl/audio/CLUS128/{split}.src", 'r') as f:
            boundaries = f.readlines()
        self.boundaries = [[int(b) for b in boundary.split()] for boundary in boundaries]
        self.boundaries = [torch.tensor(boundary) for boundary in self.boundaries]
        
        # Load features
        self.features = torch.tensor(np.load(f"/work/dcml0714/uasr_rl/rl/audio/{split}.npy"))
        # Load legnths
        with open(f"/work/dcml0714/uasr_rl/rl/audio/{split}.lengths", 'r') as f:
            lengths = f.readlines()
        self.feature_lengths = [int(length.strip()) for length in lengths]

        
    def __len__(self):
        return len(self.feature_lengths)

    def __downsample_boundary__(self, boundary):
        # Use max pooling with stride and window size = self.srto downsample the boundary
        # We need to unsqueeze because max_pool1d only accepts 3D tensors
        boundary = boundary.unsqueeze(0).unsqueeze(0).float()
        boundary = torch.nn.functional.max_pool1d(boundary, kernel_size=self.sr, stride=self.sr)
        boundary = boundary.squeeze(0).squeeze(0).long()
        return boundary
    def __downsample_feature__(self, feature):
        # Use mean pooling with stride and window size = self.srto downsample the boundary
        feature = feature.transpose(0, 1).unsqueeze(0).float()
        feature = torch.nn.functional.avg_pool1d(feature, kernel_size=self.sr, stride=self.sr)
        feature = feature.squeeze(0).transpose(0, 1)
        return feature
        

    def __getitem__(self, idx):  
        boundary = self.boundaries[idx]
        feature_length = self.feature_lengths[idx]
        if idx != 0:
            offset = sum(self.feature_lengths[:idx])
        else:
            offset = 0
        feature = self.features[offset: offset + feature_length]

        # add a gaussian kernel to smooth the labels
        # Step 1: smooth the labels
        if self.label_smoothing:
            boundary = torch.conv1d(boundary.unsqueeze(0).unsqueeze(0).float(), self.kernel.view(1, 1, -1), padding=self.kernel_size).squeeze(0).squeeze(0)
        
            # Step 2: normalize the labels to [0, 1] by min-max normalization
            boundary = (boundary - boundary.min()) / (boundary.max() - boundary.min() + 1e-8)

            # Step 3: convert the label in to probability by concatenating the label and 1-label
            boundary = torch.stack([boundary, 1-boundary], dim=1)
        
        # Downsample feature and labels
        if self.sr > 1:
            feature = self.__downsample_feature__(feature)
            boundary = self.__downsample_boundary__(boundary)
            
        
        assert len(feature) == len(boundary)
        return {
            "inputs_embeds": feature,
            "labels": boundary,
        }

from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn import CrossEntropyLoss
class LSModel(BertForTokenClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if len(labels.shape) == 3:
                # Reshape logits and labels to calculate loss
                logits_flat = logits.view(-1, self.num_labels)
                labels_flat = labels.view(-1, self.num_labels)
    
                # Calculating Cross-Entropy loss with probability distributions
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits_flat, labels_flat)
    
                # Reshape loss back to the original shape of labels
                loss = loss.view(labels.size(0), labels.size(1), -1).squeeze(-1)
    
                # Create a mask for non-padding tokens
                mask = attention_mask.float()
                #print(f"Shape of mask: {mask.shape}")
                #print(f"Shape of loss: {loss.shape}")
                # Apply mask to the loss
                loss *= mask
    
                # Average the loss over non-padding tokens
                loss = loss.sum() / mask.sum()
            elif len(labels.shape) == 2:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise ValueError(f"The shape of labels is not correct: {labels.shape}")
            
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

downsample_rate = 1
# Load the dataset
train_dataset = TokenClassificationEmbeddingsDataset(
    data_path="/work/dcml0714/uasr_rl/rl/audio", 
    split="train",
    downsample_rate= downsample_rate,
    label_smoothing=False,
)
valid_dataset = TokenClassificationEmbeddingsDataset(
    data_path="/work/dcml0714/uasr_rl/rl/audio", 
    split="valid",
    downsample_rate= downsample_rate,
)
test_dataset = TokenClassificationEmbeddingsDataset(
    data_path="/work/dcml0714/uasr_rl/rl/audio", 
    split="test",
    downsample_rate= downsample_rate,    
)


# This is just a dummy tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Initialize the data collator
data_collator = DataColloatorForTokenClassificationEmbeddings(tokenizer)

# Load the model
num_labels = 128  # Number of labels in your classification task
#config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels)
#config.hidden_size = 1024
#config.num_hidden_layers = 2
#config.num_attention_heads = 16
#model = BertForTokenClassification(config = config)
model = LSModel.from_pretrained('bert-large-uncased', num_labels=num_labels)
# only keep the first N_layer layers
model.bert.encoder.layer = model.bert.encoder.layer[:4]
model.config.num_hidden_layers = 4





# Get the evaluation metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1).reshape(-1)
    labels = labels.reshape(-1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    evaluation_strategy = "steps",
    eval_steps = 100,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()


