# Python Commands

## Train
- 4dim
```
python train.py -m naivebayes -i datasets/4dim.train.txt -o nb.4dim.model

python train.py -m logreg -i datasets/4dim.train.txt -o logreg.4dim.model

python train.py -m logreg_word2vec -i datasets/4dim.train.txt -o logreg_word2vec.4dim.model -e word2vec_embedding.wordvectors

python train.py -m naivebayes_tfidf -i datasets/4dim.train.txt -o naivebayes_tfidf.4dim.model
```
- odiya
```
python train.py -m naivebayes -i datasets/odiya.train.txt -o nb.odiya.model

python train.py -m logreg -i datasets/odiya.train.txt -o logreg.odiya.model

python train.py -m logreg_word2vec -i datasets/odiya.train.txt -o logreg_word2vec.odiya.model -e word2vec_embedding.wordvectors

python train.py -m naivebayes_tfidf -i datasets/odiya.train.txt -o naivebayes_tfidf.odiya.model
```
- products
```
python train.py -m naivebayes -i datasets/products.train.txt -o nb.products.model

python train.py -m logreg -i datasets/products.train.txt -o logreg.products.model

python train.py -m logreg_word2vec -i datasets/products.train.txt -o logreg_word2vec.products.model -e word2vec_embedding.wordvectors

python train.py -m naivebayes_tfidf -i datasets/products.train.txt -o naivebayes_tfidf.products.model
```
- questions
```
python train.py -m naivebayes -i datasets/questions.train.txt -o nb.questions.model

python train.py -m logreg -i datasets/questions.train.txt -o logreg.questions.model

python train.py -m logreg_word2vec -i datasets/questions.train.txt -o logreg_word2vec.questions.model -e word2vec_embedding.wordvectors

python train.py -m naivebayes_tfidf -i datasets/questions.train.txt -o naivebayes_tfidf.questions.model
```
## Classify
- 4dim
```
python classify.py -m nb.4dim.model -i datasets/4dim.val.test.txt -o datasets/4dim.val.pred.txt

python classify.py -m logreg.4dim.model -i datasets/4dim.val.test.txt -o datasets/4dim.val.pred.txt

python classify.py -m logreg_word2vec.4dim.model -i datasets/4dim.val.test.txt -o datasets/4dim.val.pred.txt

python classify.py -m naivebayes_tfidf.odiya.model -i datasets/4dim.val.test.txt -o datasets/4dim.val.pred.txt

```
- odiya
```
python classify.py -m nb.odiya.model -i datasets/odiya.val.test.txt -o datasets/odiya.val.pred.txt

python classify.py -m logreg.odiya.model -i datasets/odiya.val.test.txt -o datasets/odiya.val.pred.txt

python classify.py -m logreg_word2vec.odiya.model -i datasets/odiya.val.test.txt -o datasets/odiya.val.pred.txt

python classify.py -m naivebayes_tfidf.odiya.model -i datasets/odiya.val.test.txt -o datasets/odiya.val.pred.txt
```
- questions
```
python classify.py -m nb.questions.model -i datasets/questions.val.test.txt -o datasets/questions.val.pred.txt

python classify.py -m logreg.odiya.model -i datasets/odiya.val.test.txt -o datasets/odiya.val.pred.txt

python classify.py -m logreg_word2vec.odiya.model -i datasets/odiya.val.test.txt -o datasets/odiya.val.pred.txt

python classify.py -m naivebayes_tfidf.odiya.model -i datasets/odiya.val.test.txt -o datasets/odiya.val.pred.txt
```

- products
```
python classify.py -m nb.products.model -i datasets/products/val.test.txt -o datasets/products.val.pred.txt

python classify.py -m logreg.products.model -i datasets/products/val.test.txt -o datasets/products/val.pred.txt

python classify.py -m logreg_word2vec.products.model -i datasets/products/val.test.txt -o datasets/products/val.pred.txt

python classify.py -m naivebayes_tfidf.products.model -i datasets/products/val.test.txt -o datasets/products/val.pred.txt
```