# Python Commands

## Train

- 4dim
```
python train.py -u 80 -l 0.0015 -f 40 -b 64 -e 500 -E glove.6B.50d.txt -tfidf True -threshold 1 -max_features 1000 -average_emb_sentence True -i datasets/4dim/train.txt -o 4dim.model

python train-torch.py -u 30 -l 0.001 -f 100 -b 32 -e 500 -average_emb_sentence True -adam True -E glove.6B.50d.txt -i datasets/4dim/train.txt -o torch.4dim.model
```
- odiya
```
python train.py -u 10 -ul "10" -l 0.01 -f 30 -b 128 -e 200 -tfidf True -threshold 1 -max_features 2000 -m 0.9 -average_emb_sentence True -E fasttext.wiki.300d.vec -m 0.9 -i datasets/odiya/train.txt -o odia.model

python train-torch.py -u 5 -ul "5" -l 0.01 -f 30 -b 128 -e 200 -E fasttext.wiki.300d.vec -i datasets/odiya/train.txt -o torch.odia.model
```
- products
```
python train.py -u 10 -ul "10" -l 0.02 -f 40 -b 32 -e 200 -m 0.4 -tfidf True -max_features 1000 -threshold 1 -average_emb_sentence True -E glove.6B.50d.txt -i datasets/products/train.txt -o products.model

python train-torch.py -u 20 -l 0.2 -f 60 -b 256 -e 300 -average_emb_sentence True -E glove.6B.50d.txt -i datasets/products/train.txt -o torch.products.model
```
- questions
```
python train.py -u 10 -l 0.1 -f 10 -b 32 -e 200 -average_emb_sentence True -E ufvytar.100d.txt -i datasets/questions/train.txt -o questions.model

python train-torch.py -u 5 -l 0.1 -f 10 -b 32 -e 100 -E ufvytar.100d.txt -i datasets/questions/train.txt -o torch.questions.model
```

## Classify
- 4dim
```
python classify.py -m 4dim.model -i datasets/4dim/val.test.txt -o datasets/4dim/val.test.pred.txt

python classify-torch.py -m torch.4dim.model -i datasets/4dim/val.test.txt -o datasets/4dim/val.test.pred.txt
```
- odiya
```
python classify.py -m odia.model -i datasets/odiya/val.test.txt -o datasets/odiya/val.test.pred.txt

python classify-torch.py -m torch.odia.model -i datasets/odiya/val.test.txt -o datasets/odiya/val.test.pred.txt
```

- products
```
python classify.py -m products.model -i datasets/products/val.test.txt -o datasets/products/val.test.pred.txt

python classify-torch.py -m torch.products.model -i datasets/products/val.test.txt -o datasets/products/val.test.pred.txt
```

- questions
```
python classify.py -m questions.model -i datasets/questions/val.test.txt -o datasets/questions/val.test.pred.txt

python classify-torch.py -m torch.questions.model -i datasets/questions/val.test.txt -o datasets/questions/val.test.pred.txt
```