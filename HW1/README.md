# NaiveBayes

## Train
- 4dim
```
python train.py -m naivebayes -i datasets/4dim.train.txt -o nb.4dim.model
```
- odiya
```
python train.py -m naivebayes -i datasets/odiya.train.txt -o nb.odiya.model
```
- products
```
python train.py -m naivebayes -i datasets/products.train.txt -o nb.products.model
```
- questions
```
python train.py -m naivebayes -i datasets/questions.train.txt -o nb.questions.model
```
## Classify
- 4dim
```
python classify.py -m nb.4dim.model -i datasets/4dim.val.test.txt -o datasets/4dim.val.pred.txt
```
- odiya
```
python classify.py -m nb.odiya.model -i datasets/odiya.val.test.txt -o datasets/odiya.val.pred.txt
```
- questions
```
python classify.py -m nb.questions.model -i datasets/questions.val.test.txt -o datasets/questions.val.pred.txt
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