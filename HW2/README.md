# Python Commands

## Train

- 4dim
```
python train.py -u 5 -l 0.1 -f 10 -b 32 -e 100 -E glove.6B.50d.txt -i datasets/4dim/train.txt -o 4dim.model

python train.py -u 5 -l 0.1 -f 10 -b 32 -e 100 -E glove.6B.50d.txt -i datasets/4dim/train.txt -o torch.4dim.model -m torch
```
- odiya
```
python train.py -u 5 -l 0.1 -f 10 -b 32 -e 100 -E fasttext.wiki.300d.vec -i datasets/odiya/train.txt -o odiya.model

python train.py -u 5 -l 0.1 -f 10 -b 32 -e 100 -E fasttext.wiki.300d.vec -i datasets/odiya/train.txt -o torch.odiya.model -m torch
```
- products
```
python train.py -u 5 -l 0.1 -f 10 -b 32 -e 100 -E glove.6B.50d.txt -i datasets/products/train.txt -o products.model

python train.py -u 5 -l 0.1 -f 10 -b 32 -e 100 -E glove.6B.50d.txt -i datasets/products/train.txt -o torch.products.model -m torch
```
- questions
```
python train.py -u 5 -l 0.1 -f 10 -b 32 -e 100 -E ufvytar.100d.txt -i datasets/questions/train.txt -o questions.model

python train.py -u 5 -l 0.1 -f 10 -b 32 -e 100 -E ufvytar.100d.txt -i datasets/questions/train.txt -o torch.questions.model -m torch
```

## Classify
- 4dim
```
python classify.py -m 4dim.model -i datasets/4dim/val.test.txt -o datasets/4dim/val.test.pred.txt

python classify.py -m torch.4dim.model -i datasets/4dim/val.test.txt -o datasets/4dim/val.test.pred.txt
```
- odiya
```
python classify.py -m odiya.model -i datasets/odiya/val.test.txt -o datasets/odiya/val.test.pred.txt

python classify.py -m torch.odiya.model -i datasets/odiya/val.test.txt -o datasets/odiya/val.test.pred.txt
```

- products
```
python classify.py -m products.model -i datasets/products/val.test.txt -o datasets/products/val.test.pred.txt

python classify.py -m torch.products.model -i datasets/products/val.test.txt -o datasets/products/val.test.pred.txt
```
- questions
```
python classify.py -m questions.model -i datasets/questions/val.test.txt -o datasets/questions/val.test.pred.txt

python classify.py -m torch.questions.model -i datasets/questions/val.test.txt -o datasets/questions/val.test.pred.txt
```