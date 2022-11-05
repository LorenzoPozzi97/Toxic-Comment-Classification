set_seed(CONFIG.seed)

train_toxicity = train["comment_text"].map(preprocess_function)
train_labels = train['target']
test_toxicity = test["comment_text"].map(preprocess_function)
test_labels = test['toxicity']
report_every = 1

model = AutoModel.from_pretrained("distilbert-base-uncased")
bert_embdim = model.config.hidden_size
linear_model = LinearClassifier(bert_embdim, 1)
moderator = ModeratorBERT(model, linear_model).to(device)

optimizer = torch.optim.Adam(moderator.parameters(), lr=CONFIG.learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3], gamma=0.1)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

# empty lists to collect matrics
train_epoch_loss = []
train_epoch_accuracy = []
train_epoch_precision = []
train_epoch_recall = []
train_epoch_auc = []
test_epoch_loss = []
test_epoch_accuracy = []
test_epoch_precision = []
test_epoch_recall = []
test_epoch_auc = []

train_probas = torch.empty((0))
test_probas = torch.empty((0))
train_groundtruth = torch.empty((0))
test_groundtruth = torch.empty((0))

# Arbitrary thershold
t = 0.3
for epoch in trange(CONFIG.epochs, desc='Training Epochs'):
  # training
  moderator.train()
  for x, y in get_batches(train_toxicity, train_labels, CONFIG.batch_size):
    optimizer.zero_grad()
    pred = moderator(x).squeeze(1)
    loss = criterion(pred, y) 
    loss.backward()
    optimizer.step() 
  scheduler.step()

  # validation
  if (epoch + 1) % report_every == 0:
    with torch.no_grad():
      moderator.eval()

      # train set
      train_loss = 0.
      train_items = 0.
      for x_train, y_train in get_batches(train_toxicity, train_labels, CONFIG.batch_size):
        train_items+=1
        pred_train = moderator(x_train).squeeze(1)
        train_loss += criterion(pred_train, y_train).item()
        probas = torch.sigmoid(pred_train)
        train_probas = torch.cat((train_probas.cpu(), probas.cpu()))
        train_groundtruth = torch.cat((train_groundtruth, y_train.cpu()))

      # test set
      test_loss = 0.
      test_items = 0.
      for x_test, y_test in get_batches(test_toxicity, test_labels, CONFIG.batch_size):
        test_items+=1
        pred_test = moderator(x_test).squeeze(1)
        test_loss += criterion(pred_test, y_test).item()
        probas = torch.sigmoid(pred_test)
        test_probas = torch.cat((test_probas, probas.cpu()))
        test_groundtruth = torch.cat((test_groundtruth, y_test.cpu()))
  
  # Calculate the optimal theshold with Precision-Recall Curve
  p, r, train_thresholds = precision_recall_curve(train_groundtruth, train_probas)
  fscore = (2 * p * r) / (p + r)
  
  fscore[np.isnan(fscore)] = 0
  ix = np.argmax(fscore)
  train_optthr = train_thresholds[ix]
  print('(Train) Best Threshold=%f, F-Score=%.3f' % (train_thresholds[ix], fscore[ix]))

  p, r, test_thresholds = precision_recall_curve(test_groundtruth, test_probas)
  fscore = (2 * p * r) / (p + r)
  fscore[np.isnan(fscore)] = 0
  ix = np.argmax(fscore)
  test_optthr = test_thresholds[ix]
  print('(Test) Best Threshold=%f, F-Score=%.3f' % (test_thresholds[ix], fscore[ix]))

  # Print metrics at the end of the epoch
  print()      
  print('-'*30)      
  print('Train Loss:', train_loss/train_items)
  print('Test Loss:', test_loss/test_items)
  print('-'*30)  
  print('Train Accuracy:', accuracy_score(train_groundtruth, polarizer(train_probas, t)))
  print('Test Accuracy:', accuracy_score(test_groundtruth, polarizer(test_probas, t)))
  print('-'*30)  
  print('Train Precision:', precision_score(train_groundtruth, polarizer(train_probas, t)))
  print('Test Precision:', precision_score(test_groundtruth, polarizer(test_probas, t)))
  print('-'*30)  
  print('Train Recall:', recall_score(train_groundtruth, polarizer(train_probas, t)))
  print('Test Recall:', recall_score(test_groundtruth, polarizer(test_probas, t)))
  print('-'*30)  
  print('Train AUC:', roc_auc_score(train_groundtruth, train_probas))
  print('Test AUC:', roc_auc_score(test_groundtruth, test_probas))
  print('-'*30)
