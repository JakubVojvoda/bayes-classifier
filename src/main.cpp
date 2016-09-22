/**
 *
 *  Binary classification using Bayesian classifier
 *  by Jakub Vojvoda, github.com/JakubVojvoda
 *  2016
 *
 *  GNU LGPL v3 (see LICENSE)
 *  file: main.cpp
 */

#include <iostream>
#include <string>
#include <sstream>

#include "bayesclassifier.h"
#include "evaluator.h"

#define VARIANT_ERR   -1
#define VARIANT_EVAL   1
#define VARIANT_TEST   2
#define VARIANT_THRESH 3

// Command line arguments
typedef struct params {

  int variant;

  std::string train_positive;
  std::string train_negative;
  std::string test_positive;
  std::string test_negative;
  std::string test_image;

  int quantization;
  int method;
  bool subsampling;
  double threshold;

  params() {
    variant = VARIANT_ERR;
    quantization = 16;
    method = BAYESIAN_RGB;
    subsampling = false;
    threshold = -1;
  }
} params_t;

params_t parseArguments(int argc, char **argv);
void printUsage();


int main(int argc, char **argv)
{
  // Parse command line arguments
  params_t p = parseArguments(argc, argv);

  if (p.variant == VARIANT_ERR) {
    std::cerr << "Wrong format or number of arguments." << std::endl;
    printUsage();
    return 1;
  }

  // Compute threshold for each sample from training dataset and print table
  // showing false positive and true positive rate for different threshold values
  if (p.variant == VARIANT_THRESH) {

    Evaluator eval;

    // Compute thresholds for training samples
    std::vector<training_sample_t> training;
    training = eval.computeThreshold(p.train_positive, p.train_negative,
                                     p.quantization, p.method, p.subsampling);

    if (training.empty()) {
      std::cerr << "Failed to load positive or negative training samples." << std::endl;
      return 1;
    }

    std::cout << "threshold" << "\t" << "FP/(TP+FP)" << "\t" << "TP/(TP+FN)" << std::endl;
    for (double threshold = 0; threshold <= 1.0; threshold += 0.01) {
      int TP = 0, FP = 0;
      int TN = 0, FN = 0;

      // Calculate true and false classification rates
      for (unsigned int i = 0; i < training.size(); i++) {
        double prob   = training.at(i).probability;
        bool positive = training.at(i).positive;

        if ( positive && prob >  threshold) { TP++; }
        if (!positive && prob <= threshold) { TN++; }
        if ( positive && prob <= threshold) { FN++; }
        if (!positive && prob >  threshold) { FP++; }
      }

      int N = TN + FP;
      int P = TP + FN;
      std::cout << threshold << "\t" << (double)FP / N
                << "\t" << (double)TP / P << std::endl;
    }
  }

  // Evaluate method using training and test dataset
  else if (p.variant == VARIANT_EVAL) {

    if (p.threshold < 0) {
      std::cerr << "Use --threshold to define positive threshold value." << std::endl;
      printUsage();
      return 1;
    }

    BayesClassifier bayes(p.quantization, p.method, p.subsampling);
    Evaluator eval;

    if (!bayes.train(p.train_positive, p.train_negative)) {
      std::cout << p.train_positive << " " << p.train_negative << std::endl;
      std::cerr << "Failed to open training text file." << std::endl;
      return 1;
    }

    // Evaluate Bayes classifier using positive and negative image from test dataset
    double precision, recall;
    eval.evaluate(bayes, p.test_positive, p.test_negative, p.threshold, precision, recall);

    printf("Precision %.2f %% \n", precision * 100.0);
    printf("Recall %.2f %% \n", recall * 100.0);

    //printf("Elapsed training time %.3f ms\n", train_time);
    //printf("Elapsed test time %.3f ms\n", test_time);
  }

  // Calculate probability of sample (that it belongs to positive class)
  else if (p.variant == VARIANT_TEST) {

    if (p.test_image.empty()) {
      std::cerr << "Input image not found (use parameter --image)." << std::endl;
      return 1;
    }

    BayesClassifier bayes(p.quantization, p.method, p.subsampling);

    if (!bayes.train(p.train_positive, p.train_negative)) {
      std::cerr << "Failed to open training text file." << std::endl;
      return 1;
    }

    bitmap_image image(p.test_image);

    if (!image) {
      std::cerr << "Image " << p.test_image << " not found" << std::endl;
      return 1;
    }

    // Compute probability for input sample
    double probability = bayes.predict(image);
    printf("Posterior probability of sample: %.2f %% \n", probability * 100);
  }

  return 0;
}

// Print help
void printUsage()
{
  std::cout << "Usage: ./bayes variant input ..." << std::endl
    << "  variant --evaluate: evaluation of implemented method" << std::endl
    << "  variant --analyze:  show table of rates for training samples" << std::endl
    << "  variant --test:     predict probability for sample" << std::endl
    << "Required arguments:" << std::endl
    << "  evaluate: --test pos neg, --train pos neg, --threshold num" << std::endl
    << "  analyze:  --train pos neg" << std::endl
    << "  test:     --train pos neg, --image path" << std::endl
    << "Optional arguments:" << std::endl
    << "  --method BAYESIAN_R or --method BAYESIAN_RGB (default)" << std::endl
    << "  --q num: change size of histogram dimensions (default 16)" << std::endl
    << "  --subsample: subsample images to descrease exec time (default not use)" << std::endl
    << "Example:" << std::endl
    << "  image_operations.exe --evaluate --threshold 0.37 --subsample" << std::endl
    << "  image_operations.exe --evaluate --train p1.txt n1.txt --test p2.txt n2.txt --threshold 0.34" << std::endl
    << "  image_operations.exe --analyze --train p.txt n.txt" << std::endl
    << "  image_operations.exe --test --image img.bmp" << std::endl;
}

// Parse possible command line arguments
params_t parseArguments(int argc, char **argv)
{
  params_t p;
  std::string path(argv[0]);

  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    if (arg.compare("--evaluate") == 0) {
      p.variant = VARIANT_EVAL;

    } else if (arg.compare("--analyze") == 0) {
      p.variant = VARIANT_THRESH;

    } else if (arg.compare("--predict") == 0) {
      p.variant = VARIANT_TEST;

    } else if (arg.compare("--train") == 0) {
      if (argc <= i+2) { p.variant = VARIANT_ERR; break; }
      p.train_positive = std::string(argv[++i]);
      p.train_negative = std::string(argv[++i]);

    } else if (arg.compare("--test") == 0) {
      if (argc <= i+2) { p.variant = VARIANT_ERR; break; }
      p.test_positive = std::string(argv[++i]);
      p.test_negative = std::string(argv[++i]);

    } else if (arg.compare("--image") == 0) {
      if (argc <= i+1) { p.variant = VARIANT_ERR; break; }
      p.test_image = std::string(argv[++i]);

    } else if (arg.compare("--threshold") == 0) {
      if (argc <= i+1) { p.variant = VARIANT_ERR; break; }
      std::istringstream s(argv[++i]);
      s >> p.threshold;

    } else if (arg.compare("--q") == 0) {
      if (argc <= i+1) { p.variant = VARIANT_ERR; break; }
      std::istringstream s(argv[++i]);
      s >> p.quantization;

    } else if (arg.compare("--method") == 0) {
      if (argc <= i+1) { p.variant = VARIANT_ERR; break; }
      std::string m(argv[++i]);
      if (m.compare("rgb") || m.compare("RGB")) {
        p.method = BAYESIAN_RGB;
      } else if (m.compare("r") || m.compare("R")) {
        p.method = BAYESIAN_R;
      } else {
        p.variant = VARIANT_ERR;
        break;
      }

    } else if (arg.compare("--subsample") == 0) {
        p.subsampling = true;

    } else {
        p.variant = VARIANT_ERR;
        break;
    }
  }

  if (p.quantization > 256 || p.quantization <= 0 || (p.quantization & (p.quantization - 1)) != 0) {
    std::cerr << "Quantization value must be power of 2 and lower than 256." << std::endl;
    p.variant = VARIANT_ERR;
    return p;
  }

  if (p.train_positive.empty()) {p.train_positive = "../data/train_pos.txt";}
  if (p.train_negative.empty()) {p.train_negative = "../data/train_neg.txt";}
  if (p.test_positive.empty())  {p.test_positive = "../data/test_pos.txt";}
  if (p.test_negative.empty())  {p.test_negative = "../data/test_neg.txt";}

  return p;
}
