/**
 *
 *  Binary classification using Bayesian classifier
 *  by Jakub Vojvoda, github.com/JakubVojvoda
 *  2016
 *
 *  GNU LGPL v3 (see LICENSE)
 *  file: bayesclassifier.h
 */

#ifndef BAYESCLASSIFIER_H
#define BAYESCLASSIFIER_H

#include "bitmap_image.hpp"
#include "nvector.h"

#define BAYESIAN_R   1
#define BAYESIAN_RGB 3

// Implementation of Bayes classifier. The classifier is trained
// on positive and negative images of type bitmap_image and new samples
// are predicted using the pretrained model.
//
class BayesClassifier
{
public:
  // Create Bayes classifier
  //  quantization - quantize color, ie change size of histogram
  //  method_space - use only R (BAYESIAN_R)
  //     or all components (BAYESIAN_RGB) of RGB color space
  //  subsampling - subsample input data
  BayesClassifier(int quantization,
                  int method_space = BAYESIAN_RGB,
                  bool subsampling = false);

  // Train model from positive and negative samples
  bool train(std::string positive, std::string negative);
  bool train(std::vector<bitmap_image> positive,
             std::vector<bitmap_image> negative);

  // Compute probability for input sample
  double predict(bitmap_image sample);

  // Get number of used training samples
  unsigned int getTrainingSize();

protected:
  // Prior probability
  double prior;

  // Add sample to model
  void addSample(bitmap_image sample, bool positive = true);

  // Add new sample to trained model
  template <typename T, unsigned int dim>
  void addHistogram(vector<T, dim> &histogram, bitmap_image image);

private:
  int method;
  int quant;
  int subsample;

  vector1D positive1D;
  vector1D negative1D;

  vector3D positive3D;
  vector3D negative3D;

  unsigned int number_of_samples;

  unsigned int positive_samples;
  unsigned int negative_samples;
};

#endif // BAYESCLASSIFIER_H
