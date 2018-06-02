# Factorization Machine
Factorization Machine in tensorflow.  
Tested with tensorflow v1.4.1 on CPU

## Model Structure
![Model Structure](./graph.png) 

## Example
```
python train_example.py
python restore_example.py
```

## Functions
- FM Core  
  train_step(sess, input_x, input_y)
  eval_loss(sess, input_x, input_y)
  get_embedding(sess, input_x)

- FM Classifier  
  predict_proba(sess, input_x)
  eval_auc(selff, input_x, input_y)

- FM Regressor  
  predict(sess, input_x)


## Reference:
- [Factorization Machine](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
