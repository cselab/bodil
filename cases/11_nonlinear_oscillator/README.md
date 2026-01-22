# nonlinear oscillator

How to select beta when the model is mispecified?


```
./laplace_beta_train_val.py --datagen duffing --no-plot --num-data 20
./laplace_beta_train_val.py --datagen duffing --no-plot --num-data 200
./laplace_beta_train_val.py --datagen duffing --no-plot --num-data 2000
```

When it is the correct model: does beta become larger, as expected?
```
./laplace_beta_train_val.py --datagen linear --no-plot --num-data 20
./laplace_beta_train_val.py --datagen linear --no-plot --num-data 200
./laplace_beta_train_val.py --datagen linear --no-plot --num-data 2000
```
