# Bayes classifier
Binary classification of images using Bayes classifier

### Usage
There are defined 3 usage cases

1. Evaluate trained classifier on the test set 
 * `./bayes --evaluate --test p1.txt n1.txt --train p2.txt n2.txt --threshold NUM [...]` 
2. Get table which contains precision and recall for possible threshold values (computed using cross-validation) 
 * `./bayes --analyze --train pos.txt neg.txt [--q 2^NUM] [--method BAYESIAN_RGB | --method BAYESIAN_R] [--subsample]`
3. Calculate a probability for image `img.bmp` (only .bmp format supported)
 * `./bayes --predict --train pos.txt neg.txt --image img.bmp [--q 2^NUM] [--method BAYESIAN_RGB | --method BAYESIAN_R] [--subsample]`

### Command line arguments
Run `./bayes VARIANT INPUT OPTIONAL` where

* `VARIANT`
 * `--evaluate`: evaluation of implemented method
 * `--analyze`: show table of rates for training samples 
 * `--predict`: predict probability for sample using defined threshold

* `INPUT`
 * `--test positive.txt negative.txt`
 * `--train positive.txt negative.txt`

* `OPTIONAL`
 * `--method`: possible values `BAYESIAN_R` or `BAYESIAN_RGB` (default is `BAYESIAN_RGB`)
 * `--q NUM`: change size of histogram dimensions (default 16)
 * `--subsample`: subsample images to descrease exec time (default not use)

### Examples

* `./bayes --evaluate --threshold 0.37 --subsample`
* `./bayes --evaluate --train p1.txt n1.txt --test p2.txt n2.txt --threshold 0.34`
* `./bayes --analyze --train p.txt n.txt`
* `./bayes --train p1.txt n1.txt --test --image img.bmp`
