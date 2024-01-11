#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

template <typename Derived>
void pretty_print(const std::string &name, const Eigen::DenseBase<Derived> &M) {
  Eigen::IOFormat format = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "\t[", "]", "{", "}");
  std::cout << name << std::endl;
  std::cout << M.format(format) << std::endl;
}

template <typename Derived1, typename Derived2>
void pretty_print(const std::string &name, const Eigen::DenseBase<Derived1> &M, const Eigen::DenseBase<Derived2> &dM) {
  Eigen::IOFormat format = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "\t[", "]");
  std::cout << name << std::endl;
  for (int i = 0; i < M.rows(); ++i) {
    std::cout << '\t' << M.row(i).format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", "));
    std::cout << '\t' << dM.row(i).format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", "));
    std::cout << std::endl;
  }
}

namespace NeuralNetwork {

  /* Activation Functions */
  namespace Activation {
    // Identity Activation
    template <int l, int m, int n>
    void identity(const Eigen::Array<float, l, n> &Z, Eigen::Array<float, l, n> &Y, Eigen::Array<float, l, n> &df) {
      Y = Z;
      df = Eigen::Array<float, l, n>::Ones();
    }

    // Sigmoid Activation
    template <int l, int m, int n>
    void sigmoid(const Eigen::Array<float, l, n> &Z, Eigen::Array<float, l, n> &Y, Eigen::Array<float, l, n> &df) {
      Y = 1.f / (1.f + Eigen::exp(-Z));
      df = Y * (1.f - Y);
    }

    // Rectified Linear Unit (ReLU) Activation
    template <int l, int m, int n>
    void rectified(const Eigen::Array<float, l, n> &Z, Eigen::Array<float, l, n> &Y, Eigen::Array<float, l, n> &df) {
      const float alpha = 0.1f; // Leaky ReLU factor
      df = 0.5f * ((1.f - alpha) * Eigen::sign(Z) + (1.f + alpha));
      Y = df * Z;
    }
  } // namespace Activation

  /* Loss Functions */
  namespace Loss {
    // Squared L2 Norm Loss
    template <int l, int n>
    float squared(const Eigen::Array<float, l, n> &Yref, const Eigen::Array<float, l, n> &Y, Eigen::Array<float, l, n> &dY) {
      dY = Y - Yref;
      return 0.5f * dY.matrix().squaredNorm();
    }

    // Huber Loss
    template <int l, int n>
    float huber(const Eigen::Array<float, l, n> &Yref, const Eigen::Array<float, l, n> &Y, Eigen::Array<float, l, n> &dY) {
      const Eigen::Array<float, l, n> diff = Y - Yref;
      const Eigen::Array<float, l, n> sqrt = Eigen::sqrt(diff * diff + 1.f);
      dY = diff / sqrt;
      return (sqrt + 1.f).sum();
    }
  } // namespace Loss

  /* Neural Network Layer */
  template <int l, int m, int n>
  struct Layer {
    // Matrix and Array members
    Eigen::Matrix<float, m, n> weights, dWeights;
    Eigen::Matrix<float, 1, n> biases, dBiases;
    Eigen::Array<float, l, n> output, dOutput, df;

    // Activation Function Pointer
    void (*activation)(const Eigen::Array<float, l, n> &Z, Eigen::Array<float, l, n> &Y, Eigen::Array<float, l, n> &df);

    void initialise(const std::string &act) {
      weights = Eigen::Matrix<float, m, n>::Random();
      dWeights.setZero();
      biases = Eigen::Matrix<float, 1, n>::Random();
      dBiases.setZero();
      output.setZero();
      dOutput.setZero();
      df.setZero();
      // Assign the activation function based on the input string
      if (act.compare("identity") == 0)
        activation = Activation::identity<l, m, n>;
      else if (act.compare("sigmoid") == 0)
        activation = Activation::sigmoid<l, m, n>;
      else
        activation = Activation::rectified<l, m, n>;
    }

    // Print
    void print() {
      std::cout << std::scientific;
      pretty_print("A", weights, dWeights);
      pretty_print("B", biases, dBiases);
      pretty_print("Y", output, df);
    }

    // Forward Propagation
    void forward(const Eigen::Array<float, l, m> &input) {
      const Eigen::Matrix<float, l, n> Z = (input.matrix() * weights).rowwise() + biases;
      activation(Z.array(), output, df);
    }

    // Backward Propagation
    void backward(const Eigen::Array<float, l, m> &input, Eigen::Array<float, l, m> &dInput) {
      const Eigen::Matrix<float, l, n> dZ = (df * dOutput).matrix();
      dBiases = dZ.colwise().sum();
      dWeights = input.matrix().transpose() * dZ;
      dInput = (dZ * weights.transpose()).array();
    }

    // Update Weights and Biases
    void update(const float step) {
      weights -= step * dWeights;
      biases -= step * dBiases;
    }
  };
} // namespace NeuralNetwork

/*
const int l = 5, m = 2, n = 1;

int main() {
  Eigen::Array<float, l, m> X, dX;
  Eigen::Array<float, l, n> Y, dY;
  // clang-format off
  X(0, 0) = 0.f; X(0, 1) = 0.f; Y(0) = 0.f;
  X(1, 0) = 0.f; X(1, 1) = 1.f; Y(1) = 1.f;
  X(2, 0) = 1.f; X(2, 1) = 0.f; Y(2) = 1.f;
  X(3, 0) = 1.f; X(3, 1) = 1.f; Y(3) = 2.f;
  X(4, 0) = .5f; X(4, 1) = .5f; Y(4) = 1.f;
  // clang-format on
  dX = Eigen::Array<float, l, m>::Ones();

  // Example usage of the neural network layer
  float learningRate = 0.01f;
  NeuralNetwork::Layer<l, m, n> layer;
  layer.initialise("rectified");

  for (int t = 0; t < 1000; t++) {
    layer.forward(X);
    std::cout << "error : " << NeuralNetwork::Loss::squared(Y, layer.output, layer.dOutput) << std::endl;
    layer.backward(X, dX);
    // layer.print();
    layer.update(learningRate);
  }

  return 0;
}
*/

const int l = 5, m = 2, n = 1;

int main() {
  Eigen::Array<float, l, m> X, dX;
  Eigen::Array<float, l, n> Y, dY;
  // clang-format off
  X(0, 0) = 0.f; X(0, 1) = 0.f; Y(0) = 0.f;
  X(1, 0) = 0.f; X(1, 1) = 1.f; Y(1) = 1.f;
  X(2, 0) = 1.f; X(2, 1) = 0.f; Y(2) = 1.f;
  X(3, 0) = 1.f; X(3, 1) = 1.f; Y(3) = 0.f;
  X(4, 0) = .5f; X(4, 1) = .5f; Y(4) = 1.f;
  // clang-format on
  dX = Eigen::Array<float, l, m>::Ones();

  // Example usage of the neural network layer
  float learningRate = 0.01f;
  NeuralNetwork::Layer<l, m, 2> layer1;
  layer1.initialise("rectified");
  NeuralNetwork::Layer<l, 2, n> layer2;
  layer2.initialise("rectified");

  for (int t = 0; t < 10000; t++) {
    layer1.forward(X);
    layer2.forward(layer1.output);
    std::cout << "error : " << NeuralNetwork::Loss::squared(Y, layer2.output, layer2.dOutput) << std::endl;
    layer2.backward(layer1.output, layer1.dOutput);
    layer1.backward(X, dX);

    /*layer1.print();
    layer2.print();*/

    layer1.update(learningRate);
    layer2.update(learningRate);
  }

  return 0;
}

/* const int l = 4, m = 2, n = 1;

int main() {
  Eigen::Matrix<float, l, m> X, dX;
  Eigen::Array<float, l, n> Y, dY;
  // clang-format off
  X(0, 0) = 0.f; X(0, 1) = 0.f; Y(0) = 0.f;
  X(1, 0) = 0.f; X(1, 1) = 1.f; Y(1) = 1.f;
  X(2, 0) = 1.f; X(2, 1) = 0.f; Y(2) = 1.f;
  X(3, 0) = 1.f; X(3, 1) = 1.f; Y(3) = 0.f;
  // clang-format on
  dX = Eigen::Matrix<float, l, m>::Ones();

  // Example usage of the neural network layer
  float learningRate = 0.09f;
  NeuralNetwork::Layer<l, m, n> layer;
  layer.initialise("rectified");

  for (int t = 0; t < 1000; t++) {
    layer.forward(X);
    std::cout << "error : " << NeuralNetwork::Loss::squared(Y, layer.output, layer.dOutput) << std::endl;
    layer.backward(X, dX);
    layer.print();
    layer.update(learningRate);
  }

  return 0;
} */

/*

Eigen::Matrix<float, 3, 2> input = Eigen::Matrix<float, 3, 2>::Random();
  Eigen::Matrix<float, 3, 2> dInput;
  Eigen::Matrix<float, 3, 2> outputGradient = Eigen::Matrix<float, 3, 2>::Random();

*/

/*
const int l = 4, m = 2, n = 1;

int main() {
  Eigen::Matrix<float, l, m> X, dX;
  Eigen::Array<float, l, n> Y, dY;
  // clang-format off
  X(0, 0) = 0.f; X(0, 1) = 0.f; Y(0) = 1.f;
  X(1, 0) = 0.f; X(1, 1) = 1.f; Y(1) = 1.f;
  X(2, 0) = 1.f; X(2, 1) = 0.f; Y(2) = 1.f;
  X(3, 0) = 1.f; X(3, 1) = 1.f; Y(3) = 1.f;
  // clang-format on
  dX = Eigen::Matrix<float, l, m>::Ones();

  pretty_print("X", X, dX);
  pretty_print("Y", Y, dY);

  Layer<l, m, n> layer;
  layer.initialise("identity");
  layer.print();

  for (int t = 0; t < 1000; t++) {
    layer.propagation(X);
    std::cout << "error : " << squared(Y, layer.Y, layer.dY) << std::endl;
    layer.retropagation(X, dX);
    layer.update(.01f);
    layer.print();
  }
}
*/

/* const int l = 5, m = 1, n = 1;

int main() {
  Eigen::Matrix<float, l, m> X, dX = Eigen::Matrix<float, l, m>::Ones();
  Eigen::Array<float, l, n> Y, dY;
  X(0, 0) = 0.f; Y(0, 0) = 1.f;
  X(1, 0) = 1.f; Y(1, 0) = 0.f;
  X(2, 0) = 2.f; Y(2, 0) = -1.f;
  X(3, 0) = 3.f; Y(3, 0) = -2.f;
  X(4, 0) = 4.f; Y(4, 0) = -3.f;

  pretty_print("X", X, dX);
  pretty_print("Y", Y, dY);

  Layer<l, m, n> layer;
  layer.initialise("identity");
  // layer.A(0, 0) = 1.f;
  // layer.B(0) = 1.f;
  layer.print();

  for (int t = 0; t < 100; t++) {
    layer.propagation(X);
    std::cout << "error : " << squared(Y, layer.Y, layer.dY) << std::endl;
    layer.retropagation(X, dX);
    layer.update(.05f);
    layer.print();
  }
} */

/* const int l = 1, m = 1, n = 1;

int main() {
  Eigen::Matrix<float, l, m> X, dX;
  Eigen::Matrix<float, l, n> Y, dY;
  X(0, 0) = 0.f; Y(0) = 0.f; dX(0, 0) = 1.f;

  pretty_print( "X", X, dX);
  pretty_print( "Y", Y, dY);

  Layer<l, m, n> layer;
  layer.initialise("sigmoid");
  layer.A(0, 0) = 1.f;
  layer.B(0) = 1.f;
  layer.print();

  for (int t = 0; t < 1; t++) {
    layer.propagation(X);
    std::cout << "error : " << squared(layer.Y, Y, layer.dY, true) << std::endl;
    layer.retropagation(X, dX);
    // layer.update(.01f);
    layer.print();
  }
}
*/

/* const int l = 4, m = 2, n = 1;

int main() {
  Eigen::Matrix<float, l, m> X, dX;
  Eigen::Matrix<float, l, n> Y, dY;
  // clang-format off
  X(0, 0) = 0.f; X(0, 1) = 0.f; Y(0) = 1.f;
  X(1, 0) = 0.f; X(1, 1) = 1.f; Y(1) = 1.f;
  X(2, 0) = 1.f; X(2, 1) = 0.f; Y(2) = 1.f;
  X(3, 0) = 1.f; X(3, 1) = 1.f; Y(3) = 1.f;
  // clang-format on
  dX = Eigen::Matrix<float, l, m>::Ones();

  pretty_print( "X", X, dX);
  pretty_print( "Y", Y, dY);

  Layer<l, m, 2> layer1;
  layer1.initialise("identity");
  layer1.print();

  Layer<l, 2, n> layer2;
  layer2.initialise("identity");
  layer2.print();

  for (int t = 0; t < 1; t++) {
    layer1.propagation(X);
    layer2.propagation(layer1.Y);

    std::cout << "error : " << squared(Y, layer2.Y, layer2.dY) << std::endl;

    layer2.retropagation(layer1.Y, layer1.dY);
    layer1.retropagation(X, dX);

    // layer1.update(.01f);
    // layer2.update(.01f);

    layer1.print();
    layer2.print();
  }
} */

/* const int l = 1, m = 1, n = 1;

int main() {
  Eigen::Matrix<float, l, m> X, dX;
  Eigen::Matrix<float, l, n> Y, dY;
  // clang-format off
  X(0, 0) = 1.f; Y(0) = 0.f;
  // clang-format on
  dX = Eigen::Matrix<float, l, m>::Ones();

  pretty_print( "X", X, dX);
  pretty_print( "Y", Y, dY);

  Layer<l, m, 1> layer1;
  layer1.initialise("sigmoid");
  layer1.A = Eigen::Matrix<float, m, 1>::Ones();
  layer1.B = Eigen::Matrix<float, 1, 1>::Zero();
  layer1.print();

  Layer<l, 1, n> layer2;
  layer2.initialise("identity");
  layer2.A = Eigen::Matrix<float, 1, n>::Ones();
  layer2.B = Eigen::Matrix<float, 1, n>::Zero();
  layer2.print();

  for (int t = 0; t < 10; t++) {
    layer1.propagation(X);
    layer2.propagation(layer1.Y);

    std::cout << "error : " << squared(layer2.Y, Y, layer2.dY, true) << std::endl;

    layer2.retropagation(layer1.Y, layer1.dY);
    layer1.retropagation(X, dX);

    layer1.update(.1f);
    layer2.update(.1f);

    layer1.print();
    layer2.print();
  }
}
*/
