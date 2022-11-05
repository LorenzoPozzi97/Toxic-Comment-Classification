class CONFIG: 
    seed = 101
    batch_size = 50
    epochs = 6
    learning_rate = 1e-4

def set_seed(seed):
  """
    Args:

      seed [int] ~ The desired seed.
  """
  torch.manual_seed(seed)

def polarizer(probas, threshold = 0.5):
  """
    Args:

      probas [torch.tensor] ~ Torch tensor of size (batch_size).
      threshold [int] ~ Threshold value.  
    
    Purpose: given a threshold value, it polarizes the values in the vector to be 
      either 1 or 0. 
  """
  return torch.where(probas >= threshold, 1., 0.)

def preprocess_function(examples):
  """
    Args:

      examples [str] ~ Torch tensor of size (batch_size).

    Return:

      [transformers.tokenization_utils_base.BatchEncoding] ~ A BatchEncoding 
        with the following fields:
          - input_ids: List of token ids to be fed to a model.
          - attention_mask: List of indices specifying which tokens should be 
            attended to by the model 
    
    Purpose: Tokenize the input string and convert them in numerical vectors to 
      be fed to BERT. 
  """
  return tokenizer(examples, truncation=True, return_tensors="pt")

def get_batches(source_iter, target_iter, batch_size):
  """
    Args:

      source_iter [pandas.core.series.Series] ~ Pandas dataframe to be partitioned
        in batches (senteces)
      target_iter [pandas.core.series.Series] ~ Pandas dataframe to be partitioned
        in batches (target labels)
      batch_size [int] ~ Hyperparameter defininig batch size. 

    Yields:

      [tuple] ~ A tuple of two elements:
          - a [tuple] with paired ids and attention masks vectors
          - the groundtruth labels corresponding to that batch 
    
    Purpose: Generate batch from the training/test set online.  
  """
  for batch in more_itertools.chunked(zip(source_iter, target_iter), batch_size):
    x, y = zip(*batch)
    batch_ids, batch_masks = zip(*list(map(lambda t: t.values(), x)))
    max_len = max([tensor.size()[1] for tensor in batch_ids])
    batch_ids = tuple(map(lambda n: torch.cat((n, torch.zeros(1, max_len-n.size()[1])), dim=1), batch_ids))
    batch_masks = tuple(map(lambda n: torch.cat((n, torch.zeros(1, max_len-n.size()[1])), dim=1), batch_masks))

    batch_ids = torch.cat(batch_ids).long().to(device)
    batch_masks = torch.cat(batch_masks).long().to(device)
    target = polarizer(torch.tensor(y)).to(device)
      
    yield (batch_ids, batch_masks), target
