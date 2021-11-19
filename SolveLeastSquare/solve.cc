/*================================================================
*   Copyright (C) 2021 * Ltd. All rights reserved.
*
*   Editor      : VIM
*   File name   : solver.cpp
*   Author      : YunYang1994
*   Created date: 2021-11-10 19:42:38
*   Description :
*
*===============================================================*/

#include "cppoptlib/meta.h"
#include "cppoptlib/boundedproblem.h"
#include "cppoptlib/solver/lbfgsbsolver.h"
#include "cppoptlib/solver/bfgssolver.h"
#include "cppoptlib/solver/gradientdescentsolver.h"

#include <vector>
#include <time.h>
#include <Eigen/Core>
#include <iostream>



float randNumber() {
    float r = std::rand()%100/(double)101;
    return r;
}

void getAb(std::vector<Eigen::MatrixXf> &A, std::vector<Eigen::VectorXf> &b, int num_samples, int dim) {
    Eigen::MatrixXf A_;
    Eigen::VectorXf b_;

    assert(A.size() == 0);
    assert(b.size() == 0);

    for (int i=0; i<num_samples; i++) {
        b_ = Eigen::VectorXf::Zero(2);
        b_[0] = 5060  + randNumber();
        b_[1] = 12980 + randNumber();
        b.push_back(b_);
    }

    for (int i=0; i<num_samples; i++) {
        A_ = Eigen::MatrixXf::Zero(2,dim);
        for (int j=0; j<2; j++) {
            for (int k=0; k<dim; k++) {
                A_(j, k) = (j*dim + k)*10 + randNumber();
            }
        }
        A.push_back(A_);
    }
}


class GradientDescentLSMSolver {
public:
    GradientDescentLSMSolver(float lr, const std::vector<Eigen::MatrixXf> &A_, const std::vector<Eigen::VectorXf> &b_):
    learning_rate(lr) {
        num_samples = A_.size();

        for (int i=0; i<num_samples; i++) {
            A.push_back(A_[i]);
            b.push_back(b_[i]);
        }
    }


    float solveGradients(const Eigen::VectorXf &x, Eigen::VectorXf &grad) {
        int dim = x.size();
        float loss = 0.0f;
        grad = Eigen::VectorXf::Zero(dim);
        std::vector<Eigen::VectorXf> grads(num_samples);

        for (int i=0; i<num_samples; i++) {
            auto residual = A[i]*x - b[i];
            loss += residual.squaredNorm();
            grads[i] = 2 * A[i].transpose() * residual;
        }

        for (int i=0; i<num_samples; i++) {
            for (int j=0; j<dim; j++) {
                grad[j] = (grad[j]*i + grads[i][j]) / (i+1);
            }
        }
        return loss / num_samples;
    }

    void updateGradients(Eigen::VectorXf &x, const Eigen::VectorXf &grad) {
        int dim = x.size();
        for (int i=0; i<dim; i++) {
            x[i] -= learning_rate * grad[i];
        }
    }

    void optimize(Eigen::VectorXf &x) {
        int dim = x.size();
        Eigen::VectorXf grad = Eigen::VectorXf::Zero(dim);
        for (int i=0; i<500; i++) {
            auto loss = solveGradients(x, grad);
            updateGradients(x, grad);
            // 打印训练信息
            // std::cout << "epoch :" << i << " loss: " << loss << " x: " << x.transpose() << std::endl;;
        }
    }

private:
    std::vector<Eigen::MatrixXf> A;
    std::vector<Eigen::VectorXf> b;
    int num_samples;
    float learning_rate;
};

// namespace cppoptlib {
class LeastSquaresProblem: public cppoptlib::BoundedProblem<float> {
public:
    using typename cppoptlib::BoundedProblem<float>::TVector;
    LeastSquaresProblem(std::vector<Eigen::MatrixXf> &A_, std::vector<Eigen::VectorXf> &b_, int dim_) :
        cppoptlib::BoundedProblem<float>(dim_){     // dim_ 是求解 x 的维度

        dim = dim_;
        num_samples = A_.size();

        for (int i=0; i<num_samples; i++) {
            A.push_back(A_[i]);
            b.push_back(b_[i]);
        }
    }

    float value(const TVector &x) {
        float loss = 0.;
        for (int i=0; i<num_samples; i++) {
            auto residual = A[i]*x - b[i];
            loss += residual.squaredNorm();
        }
        return loss / num_samples;
    }

    void gradient(const TVector &x, TVector &grad) {
        grad = TVector::Zero(dim);

        std::vector<Eigen::VectorXf> grads(num_samples);
        for (int i=0; i<num_samples; i++) {
            auto residual = A[i]*x - b[i];
            grads[i] = 2 * A[i].transpose() * residual;
        }

        for (int i=0; i<num_samples; i++) {
            for (int j=0; j<dim; j++) {
                grad[j] = (grad[j]*i + grads[i][j]) / (i+1);
            }
        }
    }

private:
    int num_samples;
    int dim;
    std::vector<Eigen::MatrixXf> A;
    std::vector<Eigen::VectorXf> b;
};
// }  // namespace cppoptlib

// 求最小二乘法，真实的 x = [0,1,2,3,...,11]

int main() {
    int DIM = 12;
    int NUM_SAMPLES = 1000;
    Eigen::VectorXf x1  = Eigen::VectorXf::Zero(DIM);

    std::vector<Eigen::MatrixXf> A;
    std::vector<Eigen::VectorXf> b;

    getAb(A, b, NUM_SAMPLES, DIM);

    // method 1
    GradientDescentLSMSolver solver1(1e-6, A, b);

    clock_t start = std::clock();
    solver1.optimize(x1);
    clock_t end = std::clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "time:" << duration << " solve x = " << x1.transpose() << std::endl;

    // method 2
    LeastSquaresProblem::TVector x2 = LeastSquaresProblem::TVector::Zero(DIM);
    LeastSquaresProblem f(A, b, DIM);
    f.setLowerBound(LeastSquaresProblem::TVector::Zero(DIM));        // 设置下边界 >= 0

    // cppoptlib::LbfgsbSolver<LeastSquares> solver2;
    cppoptlib::BfgsSolver<LeastSquaresProblem> solver2;
    // cppoptlib::GradientDescentSolver<LeastSquares> solver2;

    auto stopCriteria = cppoptlib::Criteria<float>::defaults();
    stopCriteria.iterations = 100;
    // stopCriteria.fDelta = 1e-3;      // 设置这个可以减少迭代次数
    // stopCriteria.xDelta = 1e-3;
    // stopCriteria.gradNorm = 1e-8;
    solver2.setStopCriteria(stopCriteria);

    start = std::clock();
    solver2.minimize(f, x2);
    end = std::clock();
    duration = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "time:" << duration << " solve x = " << x2.transpose() << "\tloss:" << f(x2) << std::endl;

    return 0;
}

