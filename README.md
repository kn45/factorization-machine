# Factorization Machine
Factorization Machine in tensorflow.  
Compatible with python2 and python3.  
Tested on:  
- tensorflow v1.4.1 & v1.8.0  
- CPU & GPU  
- py2.7 & py3.6

## Model Structure
![Model Structure](graph.png)

## Example
```
python train_example.py
python restore_example.py
tensorboard --logdir=tensorboard_log/
```

## Functions
- FM Core  
  train\_step(sess, input\_x, input\_y, lr)  
  eval\_loss(sess, input\_x, input\_y)  
  eval\_metrics(sess, input\_x, input\_y)  
  get\_embedding(sess, input\_x)  
  saver(): saver for model only  
  ckpt\_saver(): saver for all the variables(including opt etc.)  

- FM Classifier  
  predict\_proba(sess, input\_x)  
  eval\_auc(sess, input\_x, input\_y)

- FM Regressor  
  predict(sess, input\_x)


## Reference:
- [Factorization Machine](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
