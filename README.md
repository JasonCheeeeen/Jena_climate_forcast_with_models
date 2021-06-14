# jena climate forcast with models(Rnn, Cnn, Dense)
The goal in this project is to find which model is suitable for forecast in "Jena climate".</br>
The models have Rnn(Lstm), Cnn, Dense

## Find the best width for training model
1. take probable width 5~50
2. train ecah model with width 5~50
3. find the max r2_score (the best performance)
4. take correspond width to train model

## Use predict data to predict future
1. predict a new data from trainned model.
2. append prediction data to our trainning dataset
3. take last same length
4. back to 1

## Result - Compute R2 value of training and test
1. Training</br> 
Rnn   : 0.966580459961591</br>
Cnn   : 0.970434208330225</br>
Dense : 0.975481037086605</br>

2. Test</br> 
Rnn   : 0.6429095118313762</br>
Cnn   : 0.7851424365003093</br>
Dense : 0.682133078239662</br>
