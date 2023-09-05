# NaiveBayes
## Train
```
python train.py -m naivebayes -i datasets/4dim.train.txt -o nb.4dim.model
```
## Classify
```
python classify.py -m nb.4dim.model -i datasets/4dim.val.txt -o datasets/4dim.val.pred.txt
```

## Evaluate

```
python score.py -t datasets/4dim.val.txt -p datasets/4dim.val.pred.txt
```

# Evaluation script

**Args:**
- "-t": path of the input file in the form <text>TAB<label>
- "-p": path of the prediction file in the form <prediction_label>
```
python score.py -t datasets/4dim.val.txt -p datasets/4dim.val.pred.txt
```