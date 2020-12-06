# Driver's Speed Prediction according to Preview Road Curvatures

...

## Getting Started

...

### Prerequisites

...

```
python
```

### Installing

...

```
python
```

## Running the Codes 

...

### Model Training

...

```
python train.py 
```

### Model Test (with driving record data & map data)

You can test the model with the following circuit data for each driver.
Driver YJ: mugello, magione, imola, spa, YYF
Driver YS: mugello, magione, imola, spa, monza 
* YYF means Korea International Circuit (full). But, please ```YYF``` as an argument when running code.
* Examples of trained models are provided which are chpt_YJ_v1 and chpt_YS_v1.

```
python test_main.py --driver YJ --vNumber 1 --epoch 30 --circuit spa --mode 0
python test_main.py -d YJ -v 1 -e 30 -c spa -m 1  # rolling prediction 
```

### Model Test (with map data only)

You can test the model on the map data below: 
mugello, magione, imola, spa, monza, nordschleife, YYF, TB
* TB means Taebaek circuit.

```
python simulation.py -d YJ -v 1 -e 30 -c spa
```

### Model Test with animation

```
python GraphUtils_mapV.py -d YJ -v 1 -e 30 -c TB
python GraphUtils_mapV_wGT.py -d YJ -v 1 -e 30 -c YYF  # plotting with ground truth data
```
