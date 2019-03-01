To train the voting model and generate a set of predictions from the
validation and test datasets first unzip the training_citations_2017.json.gz data.

Then, from the voting directory, run 
```
run_voting.bat 
```
if on windows or 
```
bash run_voting.sh
```
if on linux.

This will save a joblib file and generate a set of predictions
from the test and validation datasets. The predictions will be stored in the 
results directory. The ensemble will be saved in the current directory.  
