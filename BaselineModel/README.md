# Driver Model (Driver's speed prediction)

* **PolyFit:** $$ v_(t+k) = f(\kappa_(t+k)) $$, k=5sec, f:Polynomial.
* **PlaneFit:** $$ v_(t+k) = v_t + f(v_t, \kappa_(t+k)) $$, k=5sec, f:Plane.
* **PlaneFit with NN:** v_(t+k) = v_t + f(v_t, \kappa_(t+1:t+k)) $$, k=5sec, f:Neural Network.

## Getting Started
To run polyFit and planeFit,
```python
main.py  
```

To run planeFit with NN,
```python
mainNN.py
```
