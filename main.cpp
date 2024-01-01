#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>

float huberLoss(float x) {
    return std::sqrt(x * x + 1.0f) - 1.0f;
}

float hberLossDerivative(float x) {
    return x / std::sqrt(x * x + 1.0f);
}

template<int l, int m, int n> // dimensions
struct Layer {
    // Eigen::Matrix<float, l, m> X, dX;
    Eigen::Matrix<float, m, n> A, dA;
    Eigen::Matrix<float, 1, n> B, dB;
    Eigen::Matrix<float, l, n> Y, dY;
    // real numeric activation function and derivative
    float (*f)(float);
    float (*df)(float);

    void print(std::ostream& os)
    {
        os << "A :\n" << A << std::endl;
        os << "dA :\n" << dA << std::endl;
        os << "B :\n" << B << std::endl;
        os << "dB :\n" << dB << std::endl;
        os << "Y :\n" << Y << std::endl;
        os << "dY :\n" << dY << std::endl;
    }

    void initialise(float (*f)(float), float (*df)(float)) {
        A = Eigen::Matrix<float, m, n>::Random();
        dA.setZero();
        B = Eigen::Matrix<float, 1, n>::Random();
        dB.setZero();
        Y.setZero();
        dY.setZero();
        this->f = f;
        this->df = df;
    }

    void propagation(const Eigen::Matrix<float, l, m> &X) {
        Y = (X * A).rowwise() + B; // for now this is a Z matrix
        dY = Y.array().exp(); // (&f);
        Y = Y.array().exp(); // (&df); // now activation is applied
    }

    void retropagation(const Eigen::Matrix<float, l, m> &X, Eigen::Matrix<float, l, m> &dX) {
        // Supposes that dY has been updated correctly, and propagates this assumption
        dB = dY.colwise().sum();
        dA = X.transpose() * dY;
        // dX *= dY * A.transpose(); // not to forget 
    }
};

const int l = 5, m = 3, n = 1;
 
int main()
{
    Eigen::Matrix<float, l, m> X, dX;
    X.setZero();
    dX.setZero();
    std::cout << "X : " << X << std::endl;
    std::cout << "dX : " << dX << std::endl;

    Layer<l, m, n> layer;
    layer.initialise(huberLoss, hberLossDerivative);
    layer.print(std::cout);

    layer.propagation(X);
    layer.print(std::cout);

    layer.retropagation(X, dX);
    layer.print(std::cout);
}
