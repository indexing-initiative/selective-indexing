To generate results, run 
```
python generate_results.py
```

This will produce a text file and the three figures presented in the paper.
The text file consists of result metrics for the ensemble, cnn, and the combined model.
Recall and precision are calculated as the decision threshold is iteratively changed,
until a recall level of 99.8% is reached. 
The initial threshold levels for each model were chosen by trial and error, 
in order to be able to visualize a similar range of recall between the models. 

