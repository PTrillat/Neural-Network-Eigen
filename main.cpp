#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>

template<int l, int m, int n> // dimensions
struct Layer {
    // Eigen::Matrix<float, l, m> X, dX;
    Eigen::Matrix<float, m, n> A, dA;
    Eigen::Matrix<float, 1, n> B, dB;
    Eigen::Matrix<float, l, n> Y, dY;
    // activation function + derivative

    void initialise() {
        A = Eigen::Matrix<float, m, n>::Random();
        dA.setZero();
        B = Eigen::Matrix<float, 1, n>::Random();
        dB.setZero();
        Y = Eigen::Matrix<float, l, n>::Random();
        dY.setZero();
    }

    void propagation(const Eigen::Matrix<float, l, m> &X) {
        Y = (X * A).rowwise() + B; // for now this is a Z matrix
        // dY = Y.array().df()
        // Y = Y.array().exp()
    }

    void retropagation(const Eigen::Matrix<float, l, m> &X, Eigen::Matrix<float, l, m> &dX) {
        // Supposes that dY has been updated correctly, and propagates this accordingly
        dB = dY.colwise().sum();
        dA = X.transpose() * dY;
        // dX (from the previous layer) contrains f'(Z) and d(Cost)/dZ = d(cost)/dY * dY/dZ
        // dX *= dY * A.transpose();
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
    layer.initialise();
    std::cout << "A : " << layer.A << std::endl;
    std::cout << "dA : " << layer.dA << std::endl;
    std::cout << "B : " << layer.B << std::endl;
    std::cout << "dB : " << layer.dB << std::endl;
    std::cout << "Y : " << layer.Y << std::endl;
    std::cout << "dY : " << layer.dY << std::endl;

    layer.propagation(X);
    std::cout << "A : " << layer.A << std::endl;
    std::cout << "dA : " << layer.dA << std::endl;
    std::cout << "B : " << layer.B << std::endl;
    std::cout << "dB : " << layer.dB << std::endl;
    std::cout << "Y : " << layer.Y << std::endl;
    std::cout << "dY : " << layer.dY << std::endl;

    layer.retropagation(X, dX);
    std::cout << "A : " << layer.A << std::endl;
    std::cout << "dA : " << layer.dA << std::endl;
    std::cout << "B : " << layer.B << std::endl;
    std::cout << "dB : " << layer.dB << std::endl;
    std::cout << "Y : " << layer.Y << std::endl;
    std::cout << "dY : " << layer.dY << std::endl;
}
