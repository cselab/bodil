# Error models for brain imaging

* `gliodil.py`: original gliodil implementation, compute the probability from loss.
* `binomial_sigmoid.py `: binomial distribution, parameter p depends on concentration (sigmoid).
* `relu.py `: approximation for `binomial_sigmoid.py`
* `relu_sq.py `: square of the relu loss (nothing physically based)
* `binomial_edema.py`: adapts `binomial_sigmoid.py` for edema region with 2 thresholds.
