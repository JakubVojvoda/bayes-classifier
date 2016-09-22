/**
 *
 *  Binary classification using Bayesian classifier
 *  by Jakub Vojvoda, github.com/JakubVojvoda
 *  2016
 *
 *  GNU LGPL v3 (see LICENSE)
 *  file: nvector.h
 */

#ifndef NVECTOR_H
#define NVECTOR_H

#include <vector>
#include <math.h>
#include <assert.h>

#define NORM_SUM 1
#define NORM_MAX 2

#ifndef NDEBUG
  #define NDEBUG
#endif

// This class represents vector in 1D, 2D or 3D space.
// Elements are stored in std::vector of size d^1, d^2 or d^3.
//
template <typename T, unsigned int dim>
class vector {
public:
  vector(unsigned int d=0, T const & t=T()) :
    d(d), data((int)std::pow((double)d, (int)dim), t) {}

  vector(const std::vector<T> t, unsigned int d) :
    d(d), data(t) {}

  // Access element
  // Returns a reference to the element at specific position
  T & operator()(unsigned int i, unsigned int j=0, unsigned int k=0) {
    assert(i + j*d + k*d*d < std::pow((double)d, (int)dim));
    assert(i < d && j < d && k < d);
    return data[i + j*d + k*d*d];
  }

  // Access element at specific position
  T const & operator()(unsigned int i, unsigned int j=0, unsigned int k=0) const {
    assert(i + j*d + k*d*d < std::pow((double)d, (int)dim));
    assert(i < d && j < d && k < d);
    return data[i + j*d + k*d*d];
  }

  // Assign new contents to the vector
  vector<T, dim> & operator=(const vector<T, dim> & src) {
    d = src.d;
    data = src.data;
    return *this;
  }

  // Insert element into specific position
  void assign(T const & t, unsigned int i, unsigned int j=0, unsigned int k=0) {
    assert(i + j*d + k*d*d < std::pow((double)d, (int)dim));
    assert(i < d && j < d && k < d);
    data[i + j*d + k*d*d] = t;
  }

  // Increment element on specific position
  void inc(unsigned int i, unsigned int j=0, unsigned int k=0) {
    assert(i + j*d + k*d*d < std::pow((double)d, (int)dim));
    assert(i < d && j < d && k < d);
    data[i + j*d + k*d*d] += 1;
  }

  // Get sum of elements in vector
  double sum() {
    double s = 0;

    for (unsigned int i = 0; i < data.size(); i++) {
      s += data.at(i);
    }
    return s;
  }

  // Get maximum value in vector
  double max() {
    return *std::max_element(data.begin(), data.end());
  }

  // Normalize vector using sum of all elements (NORM_SUM)
  // or maximum value (NORM_MAX)
  void normalize(int method = NORM_SUM) {
    double norm = (method == NORM_SUM) ? sum() : max();

    for (unsigned int i = 0; i < data.size(); i++) {
      data.at(i) /= norm;
    }
  }

  std::size_t dimension() { return d; }
  std::size_t size() { return data.size(); }

  std::vector<T> get() {
    return data;
  }

private:
  unsigned int d;
  std::vector<T> data;
};

typedef vector<unsigned long, 3> vector3UL;
typedef vector<unsigned long, 2> vector2UL;
typedef vector<unsigned long, 1> vector1UL;

typedef vector<unsigned int, 3> vector3UI;
typedef vector<unsigned int, 2> vector2UI;
typedef vector<unsigned int, 1> vector1UI;

typedef vector<double, 3> vector3D;
typedef vector<double, 2> vector2D;
typedef vector<double, 1> vector1D;

#endif // NVECTOR_H
