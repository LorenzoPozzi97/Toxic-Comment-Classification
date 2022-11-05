tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
class LinearClassifier(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LinearClassifier, self).__init__()
         self.linear = torch.nn.Sequential(
             torch.nn.Linear(input_dim, input_dim),
             torch.nn.ReLU(),
             torch.nn.Dropout(0.5),
             torch.nn.Linear(input_dim, output_dim))
         
     def forward(self, x):
         outputs = self.linear(x)
         return outputs

class ModeratorBERT(torch.nn.Module):
  def __init__(self, bert_model, linear_classifier):
    super().__init__()
    self.linear_classifier = linear_classifier
    self.bert_model = bert_model

  def forward(self, inputs):
    bert_output = self.bert_model(input_ids=inputs[0], 
                                  attention_mask=inputs[1])
    reg_features = bert_output.last_hidden_state[:,0,:]
    reg_output = self.linear_classifier(reg_features)
    return reg_output
