#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

template <int m, int n> // dimensions
void pretty_print(std::ostream &os, const std::string name,
                  const Eigen::Matrix<float, m, n> &M,
                  const Eigen::Matrix<float, m, n> &dM) {
  os << name << std::endl;
  for (int i = 0; i < m; i++) {
    os << '\t' << M(i, 0);
    for (int j = 1; j < n; j++)
      os << ", " << M(i, j);
    os << '\t' << dM(i, 0);
    for (int j = 1; j < n; j++)
      os << ", " << dM(i, j);
    os << std::endl;
  }
}

/* sigmoid activation bloc */
template <int l, int m, int n>
void sigmoid(Eigen::Matrix<float, l, n> &Z, Eigen::Matrix<float, l, n> &dZ) {
  Eigen::Array<float, l, n> sigma = 1.f / (1.f + Eigen::exp(-Z.array()));
  Z = sigma.matrix();
  dZ = (sigma * (1.f - sigma)).matrix();
}

/* relu activation bloc */
template <int l, int m, int n>
void rectified(Eigen::Matrix<float, l, n> &Z, Eigen::Matrix<float, l, n> &dZ) {
  const float alpha = .01f; // how leaky it is
  Eigen::Array<float, l, n> sigma =
      .5f * ((1.f - alpha) * Eigen::sign(Z.array()) +
             (1.f + alpha)); // leaky positive
  dZ = sigma.matrix();
  Z.array() = sigma * dZ.array();
}

/* Squared L2 norm loss */
template <int l, int n>
float squared(const Eigen::Matrix<float, l, n> &Y,
              const Eigen::Matrix<float, l, n> &Yref,
              Eigen::Matrix<float, l, n> &dY,
              bool silent) {
    const Eigen::Matrix<float, l, n> diff = Y - Yref;
    dY.array() *= diff.array();
    if (silent)
        return 0.f;
    return .5f * diff.squaredNorm();

}

/* Huber Loss bloc */
float huberLoss(float x) { return std::sqrt(x * x + 1.0f) - 1.0f; }

float huberLossDerivative(float x) { return x / std::sqrt(x * x + 1.0f); }


template <int l, int m, int n> // dimensions
struct Layer {
  // Eigen::Matrix<float, l, m> X, dX;
  Eigen::Matrix<float, m, n> A, dA;
  Eigen::Matrix<float, 1, n> B, dB;
  Eigen::Matrix<float, l, n> Y, dY;
  // compute activation function and derivative
  // should modify the inputs arrays inplace
  void (*activation)(Eigen::Matrix<float, l, n> &Z,
                     Eigen::Matrix<float, l, n> &dZ);

  void print(std::ostream &os) const {
    os << std::scientific;
    pretty_print(os, "A", A, dA);
    pretty_print(os, "B", B, dB);
    pretty_print(os, "Y", Y, dY);
  }

  void initialise(const char *act) {
    A = Eigen::Matrix<float, m, n>::Random();
    B = Eigen::Matrix<float, 1, n>::Random();
    Y.setZero();
    dA.setZero();
    dB.setZero();
    dY.setZero();
    if (act == "sigmoid")
      this->activation = sigmoid<l, m, n>;
    else
      this->activation = rectified<l, m, n>;
  }

  void propagation(const Eigen::Matrix<float, l, m> &X) {
    Y = (X * A).rowwise() + B; // for now this is a Z matrix
    activation(Y, dY);         // element wise operations
  }

  void retropagation(const Eigen::Matrix<float, l, m> &X,
                     Eigen::Matrix<float, l, m> &dX) {
    // Supposes that dY constains activation derivative
    // and computes full derivative based on this assumption
    dB = dY.colwise().sum();
    dA = X.transpose() * dY;
    dX.array() *= (dY * A.transpose()).array(); // element wise multiplication
  }

  void update(const float step) {
    A -= step * dA;
    B -= step * dB;
  }
};

const int l = 4, m = 2, n = 1;

int main() {
  Eigen::Matrix<float, l, m> X, dX;
  Eigen::Matrix<float, l, n> Y, dY;
  X(0, 0) = 0.f; X(0, 1) = 0.f; Y(0) = 1.f;
  X(1, 0) = 0.f; X(1, 1) = 1.f; Y(1) = 1.f;
  X(2, 0) = 1.f; X(2, 1) = 0.f; Y(2) = 1.f;
  X(3, 0) = 1.f; X(3, 1) = 1.f; Y(3) = 0.f;

  pretty_print(std::cout, "X", X, dX);
  pretty_print(std::cout, "Y", Y, dY);


  Layer<l, m, 2> layer1;
  Layer<l, 2, n> layer2;
  
  layer1.initialise("sigmoid");
  layer2.initialise("sigmoid");
  
  layer1.print(std::cout);
  layer2.print(std::cout);

  for (int epoch = 0; epoch < 10; epoch++) {
    for (int batch = 0; batch < 1000; batch++) {
        layer1.propagation(X);
        layer2.propagation(layer1.Y);
        
        squared(layer2.Y, Y, layer2.dY, true);
        
        layer2.retropagation(layer1.Y, layer1.dY);
        layer1.retropagation(X, dX);
        
        layer1.update(.01f);
        layer2.update(.01f);
      }
    std::cout << "error : " << squared(layer2.Y, Y, layer2.dY, false) << std::endl;
  }
  layer1.print(std::cout);
  layer2.print(std::cout);
}
