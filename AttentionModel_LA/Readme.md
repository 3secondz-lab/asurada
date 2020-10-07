## Model
* Model_LA 
    * Model leanrs local attention point by itself
* Model_LA_TS
    * Model leanrs local attention point considering a guide (spatial information, 예상 위치)
    * Takes a lot of time to learn
* Model_LA_TS2
    * Model uses a guide (spatial information) as local attention point

## Train
* Model_LA
    ```python train_LA.py```
* Model_LA_TS 
    ```python train2_GAL.py```
* Model_LA_TS2
    ```python train3_ALA.py```

You can specified training and test datasets within ```train*_*.py``` as follows.
``` circuit_tr = ['mugello'] ```
``` circuit_tr = ['mugello', 'imola', 'spa'] ```

## Test (Visualization)
Before doing the test (simulation), you can set the simulation environment parameters in ```test_constants.py```.
Example code for simulation:
``` python test_main.py -d YJ -v 1 -e 50 - c mugello -m 0 ```

