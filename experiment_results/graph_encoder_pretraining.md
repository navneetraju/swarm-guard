# Graph Pre Training Experiment Results

## Hyperparameter Search

Best config found:

```json
{
  'hidden_channels': 256,
  'dropout': 0.15725230884190566,
  'lr': 0.007025161708454167,
  'weight_decay': 0.00021170816422802345,
  'batch_size': 64
}

```

## Training

```
Epoch 1, Train Loss: 0.4503, Train Acc: 0.7825
Saved best model with val_acc = 0.7712
Epoch 2, Train Loss: 0.3177, Train Acc: 0.8648
Saved best model with val_acc = 0.8024
Epoch 3, Train Loss: 0.2703, Train Acc: 0.8891
Saved best model with val_acc = 0.9029
Epoch 4, Train Loss: 0.2736, Train Acc: 0.8891
EarlyStopping counter: 1 / 10
Epoch 5, Train Loss: 0.3162, Train Acc: 0.8839
EarlyStopping counter: 2 / 10
Epoch 6, Train Loss: 0.2949, Train Acc: 0.8795
EarlyStopping counter: 3 / 10
Epoch 7, Train Loss: 0.2304, Train Acc: 0.9064
EarlyStopping counter: 4 / 10
Epoch 8, Train Loss: 0.2043, Train Acc: 0.9203
Saved best model with val_acc = 0.9081
Epoch 9, Train Loss: 0.2062, Train Acc: 0.9263
Saved best model with val_acc = 0.9151
Epoch 10, Train Loss: 0.2563, Train Acc: 0.9064
Saved best model with val_acc = 0.9168
Epoch 11, Train Loss: 0.2083, Train Acc: 0.9133
EarlyStopping counter: 1 / 10
Epoch 12, Train Loss: 0.1871, Train Acc: 0.9263
EarlyStopping counter: 2 / 10
Epoch 13, Train Loss: 0.1684, Train Acc: 0.9333
EarlyStopping counter: 3 / 10
Epoch 14, Train Loss: 0.2444, Train Acc: 0.9081
EarlyStopping counter: 4 / 10
Epoch 15, Train Loss: 0.1949, Train Acc: 0.9255
EarlyStopping counter: 5 / 10
Epoch 16, Train Loss: 0.1670, Train Acc: 0.9376
EarlyStopping counter: 6 / 10
Epoch 17, Train Loss: 0.1684, Train Acc: 0.9315
EarlyStopping counter: 7 / 10
Epoch 18, Train Loss: 0.1821, Train Acc: 0.9376
EarlyStopping counter: 8 / 10
Epoch 19, Train Loss: 0.1221, Train Acc: 0.9541
Saved best model with val_acc = 0.9255
Epoch 20, Train Loss: 0.1300, Train Acc: 0.9471
Saved best model with val_acc = 0.9324
Epoch 21, Train Loss: 0.1805, Train Acc: 0.9324
EarlyStopping counter: 1 / 10
Epoch 22, Train Loss: 0.1546, Train Acc: 0.9411
EarlyStopping counter: 2 / 10
Epoch 23, Train Loss: 0.1689, Train Acc: 0.9333
EarlyStopping counter: 3 / 10
Epoch 24, Train Loss: 0.3126, Train Acc: 0.8752
EarlyStopping counter: 4 / 10
Epoch 25, Train Loss: 0.2263, Train Acc: 0.9038
EarlyStopping counter: 5 / 10
Epoch 26, Train Loss: 0.3109, Train Acc: 0.8752
EarlyStopping counter: 6 / 10
Epoch 27, Train Loss: 0.1902, Train Acc: 0.9376
EarlyStopping counter: 7 / 10
Epoch 28, Train Loss: 0.2279, Train Acc: 0.9133
EarlyStopping counter: 8 / 10
Epoch 29, Train Loss: 0.1749, Train Acc: 0.9350
EarlyStopping counter: 9 / 10
Epoch 30, Train Loss: 0.3184, Train Acc: 0.8830
EarlyStopping counter: 10 / 10
```

## Evaluation

Final test acc with best config = 0.9234

### Classification Report

```
              precision    recall  f1-score   support

        Real       0.93      0.91      0.92      2018
        Fake       0.91      0.93      0.92      2029

    accuracy                           0.92      4047
   macro avg       0.92      0.92      0.92      4047
weighted avg       0.92      0.92      0.92      4047
```