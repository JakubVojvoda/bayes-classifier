/**
 *
 *  Binary classification using Bayesian classifier
 *  by Jakub Vojvoda, github.com/JakubVojvoda
 *  2016
 *
 *  GNU LGPL v3 (see LICENSE)
 *  file: evaluator.cpp
 */

#include "evaluator.h"

Evaluator::Evaluator()
{
}

bool Evaluator::evaluate(BayesClassifier bayes, std::string positive_path, std::string negative_path,
                         double threshold, double &precision, double &recall)
{
  // Read test images
  if (!readSamples(positive_path, negative_path, &test_positive, &test_negative)) {
    return false;
  }

  // Compute true positive, true negative,
  // false positive and false negative rate
  int TP = 0, TN = 0;
  int FP = 0, FN = 0;

  for (unsigned int i = 0; i < test_positive.size(); i++) {
    double prob = bayes.predict(test_positive.at(i));

    if (prob >  threshold) { TP++; }
    if (prob <= threshold) { FN++; }
  }

  for (unsigned int i = 0; i < test_negative.size(); i++) {
    double prob = bayes.predict(test_negative.at(i));

    if (prob <= threshold) { TN++; }
    if (prob >  threshold) { FP++; }
  }

  // Calculate precision and recall
  precision = (double)TP / (TP + FN);
  recall = (double)TP / (TP + FP);
  return true;
}

std::vector<training_sample_t> Evaluator::computeThreshold(std::string positive_path, std::string negative_path,
                                                           int quantization, int method, bool subsampling)
{
  if (!readSamples(positive_path, negative_path, &train_positive, &train_negative)) {
    std::cerr << "Failed to open file " << positive_path << " or " << negative_path << "." << std::endl;
    return std::vector<training_sample_t>();
  }

  std::vector<training_sample_t> samples;

  // Select one positive sample from trainig dataset, train classifier using
  // other training samples and compute threshold for choosen one
  for (unsigned int i = 0; i < train_positive.size(); i++) {
    training_sample_t test = training_sample(0.0, true);
    bitmap_image test_from_train_image = train_positive.at(i);

    std::vector<bitmap_image> train(train_positive);
    train.erase(train.begin() + i);

    BayesClassifier bayes(quantization, method, subsampling);
    bayes.train(train, train_negative);
    test.probability = bayes.predict(test_from_train_image);

    samples.push_back(test);
  }

  // Select one negative sample from trainig dataset, train classifier using
  // other training samples and compute threshold for choosen one
  for (unsigned int i = 0; i < train_negative.size(); i++) {
    training_sample_t test = training_sample(0.0, false);
    bitmap_image test_from_train_image = train_negative.at(i);

    std::vector<bitmap_image> train(train_negative);
    train.erase(train.begin() + i);

    BayesClassifier bayes(quantization, method, subsampling);
    bayes.train(train_positive, train);
    test.probability = bayes.predict(test_from_train_image);

    samples.push_back(test);
  }

  return samples;
}

bool Evaluator::readSamples(std::string positive_path, std::string negative_path,
                            std::vector<bitmap_image> *positive, std::vector<bitmap_image> *negative)
{
  std::string image_path;
  std::ifstream input_positive(positive_path.c_str());
  std::ifstream input_negative(negative_path.c_str());

  if (!input_positive.is_open() || !input_negative.is_open()) {
    return false;
  }

  // Read all positive samples
  while (std::getline(input_positive, image_path)) {
    bitmap_image image(image_path.c_str());

    if (!image) {
      std::cerr << "Image " << image_path << " not found" << std::endl;
    } else {
      positive->push_back(image);
    }
  }

  // Read all negative samples
  while (std::getline(input_negative, image_path)) {
    bitmap_image image(image_path.c_str());

    if (!image) {
      std::cerr << "Image " << image_path << " not found" << std::endl;
    } else {
      negative->push_back(image);
    }
  }

  return true;
}
