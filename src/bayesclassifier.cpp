/**
 *
 *  Binary classification using Bayesian classifier
 *  by Jakub Vojvoda, github.com/JakubVojvoda
 *  2016
 *
 *  GNU LGPL v3 (see LICENCE)
 *  file: bayesclassifier.cpp
 */

#include "bayesclassifier.h"

BayesClassifier::BayesClassifier(int quantization, int method_space, bool subsampling)
{
  method = method_space;
  quant = quantization;

  if (method == BAYESIAN_RGB) {
    positive3D = vector3D(256 / quant, 0);
    negative3D = vector3D(256 / quant, 0);
  } else {
    positive1D = vector1D(256 / quant, 0);;
    negative1D = vector1D(256 / quant, 0);;
  }

  positive_samples = 0;
  negative_samples = 0;

  subsample = (subsampling) ? 2 : 1;
}


bool BayesClassifier::train(std::string positive, std::string negative)
{
  std::string image_path;
  std::ifstream input_positive(positive.c_str());
  std::ifstream input_negative(negative.c_str());

  number_of_samples = 0;

  if (!input_positive.is_open() || !input_negative.is_open()) {
    return false;
  }

  if (quant <= 0  || quant > 256 || (quant & (quant - 1)) != 0) {
    std::cerr << "Quantization value must be power of 2 and lower than 256." << std::endl;
    return false;
  }

  // Load positive images and update model
  while (std::getline(input_positive, image_path)) {
    bitmap_image image(image_path.c_str());

    if (!image) {
      std::cerr << "Image " << image_path << " not found" << std::endl;
    } else {
      addSample(image, true);
      number_of_samples++;
    }
  }

  // Load negative images and update model
  while (std::getline(input_negative, image_path)) {
    bitmap_image image(image_path.c_str());

    if (!image) {
      std::cerr << "Image " << image_path << " not found." << std::endl;
    } else {
      addSample(image, false);
      number_of_samples++;
    }
  }

  // Compute prior probability
  prior = (double) positive_samples / (positive_samples + negative_samples);

  positive1D.normalize();
  negative1D.normalize();
  positive3D.normalize();
  negative3D.normalize();

  return true;
}

bool BayesClassifier::train(std::vector<bitmap_image> positive, std::vector<bitmap_image> negative)
{
  if (quant <= 0 || (quant & (quant - 1)) != 0) {
    std::cerr << "Quantization value must be power of 2" << std::endl;
    return false;
  }

  // Get positive samples and update model
  for (unsigned int i = 0; i < positive.size(); i++) {
    addSample(positive.at(i), true);
  }

  // Get negative samples and update model
  for (unsigned int i = 0; i < negative.size(); i++) {
    addSample(negative.at(i), false);
  }

  // Compute prior probability
  prior = (double) positive_samples / (positive_samples + negative_samples);

  number_of_samples = positive.size() + negative.size();

  positive1D.normalize();
  negative1D.normalize();
  positive3D.normalize();
  negative3D.normalize();

  return true;
}

double BayesClassifier::predict(bitmap_image sample)
{
  const unsigned int height = sample.height();
  const unsigned int width  = sample.width();

  unsigned char r, g, b;
  double prob = 0;

  // Classify each pixel of input image
  for (std::size_t y = 0; y < height; y += subsample) {
    for (std::size_t x = 0; x < width; x += subsample) {

      sample.get_pixel(x, y, r, g, b);

      // Quantization
      r /= quant;
      g /= quant;
      b /= quant;

      // Compute P(x|w) and P(x|-w)
      double positive = (method == BAYESIAN_R) ? positive1D(r) : positive3D(r, g, b);
      double negative = (method == BAYESIAN_R) ? negative1D(r) : negative3D(r, g, b);

      // Compute evidence P(x) = P(x|w)P(w) + P(x|-w)P(-w)
      double evidence = prior * positive + (1 - prior) * negative;
      evidence = (evidence > 0) ? evidence : evidence + 0.00001;

      // Compute posterior probability P(w|x)
      double posterior = (positive * prior) / evidence;

      prob += posterior;
    }
  }

  // Return average posterior probability
  return prob / (((double)width / subsample) * ((double)height / subsample));
}

unsigned int BayesClassifier::getTrainingSize()
{
  return number_of_samples;
}

void BayesClassifier::addSample(bitmap_image sample, bool positive)
{
  if (positive) {
    if (method == BAYESIAN_RGB) {
      addHistogram(positive3D, sample);
    } else {
      addHistogram(positive1D, sample);
    }
    positive_samples++;
  } else {
    if (method == BAYESIAN_RGB) {
      addHistogram(negative3D, sample);
    } else {
      addHistogram(negative1D, sample);
    }
    negative_samples++;
  }
}

template <typename T, unsigned int dim>
void BayesClassifier::addHistogram(vector<T, dim> &histogram, bitmap_image image)
{
  const unsigned int height = image.height();
  const unsigned int width  = image.width();

  unsigned char r, g, b;

  // Compute histogram
  for (std::size_t y = 0; y < height; y += subsample) {
    for (std::size_t x = 0; x < width; x += subsample) {

      image.get_pixel(x, y, r, g, b);

      if (dim == 1) {
        histogram.inc(r/quant);
      } else {
        histogram.inc(r/quant, g/quant, b/quant);
      }
    }
  }
}

