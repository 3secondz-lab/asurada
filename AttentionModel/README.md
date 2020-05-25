# Driver Model (Driver's speed prediction)

* **LSTM with Attention**

## Getting Started
For preprocessing of driving dataset,
```python
create_input_files.py  
```

For model training (training/validation),
```python
train.py
```

For model evaluation,
```python
eval.py
eval.py -v True
```

## Performance (Predict 10 seconds during ahead in 0.5-second intervals using preview image)
* Train: MAPE 6.08  RMSE 8.00
* Valid: MAPE 8.06  RMSE 10.66
* Test : MAPE 8.98  RMSE 11.37
