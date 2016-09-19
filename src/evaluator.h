/**
 *
 *  Binary classification using Bayesian classifier
 *  by Jakub Vojvoda, github.com/JakubVojvoda
 *  2016
 *
 *  GNU LGPL v3 (see LICENCE)
 *  file: evaluator.h
 */

#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <vector>
#include "bayesclassifier.h"
#include "bitmap_image.hpp"

typedef struct training_sample {

  double probability;
  bool positive;

  training_sample(double prob, bool pos)
    : probability(prob), positive(pos) {}

} training_sample_t;

// Evaluation of implemented Bayes classifier
class Evaluator
{
public:
  Evaluator();

  // Evaluate Bayes classifier using positive and negative image of test dataset
  bool evaluate(BayesClassifier bayes, std::string positive_path, std::string negative_path,
                double threshold, double &precision, double &recall);

  // Compute threshold for each sample from training dataset
  std::vector<training_sample_t> computeThreshold(std::string positive_path, std::string negative_path,
                                                  int quantization, int method, bool subsampling);

protected:
  // Read defined positive and negative samples
  bool readSamples(std::string positive_path, std::string negative_path,
                   std::vector<bitmap_image> *positive, std::vector<bitmap_image> *negative);

private:
  std::vector<bitmap_image> train_positive;
  std::vector<bitmap_image> train_negative;

  std::vector<bitmap_image> test_positive;
  std::vector<bitmap_image> test_negative;

};

#endif // EVALUATOR_H
