// In the name of Allh

#include <Cross_Entropy.h>

double CrossEntropyError::calculate_error(Eigen::MatrixXd &prediction, Eigen::MatrixXd &actual) {
    std::cout << "Entered in the Cross Entropy" << std::endl ;

    double min_value = 1e-10 ;
    double max_value = 1 - 1e-10 ;

    auto correct_probs = select_correct_probs(prediction , actual) ;
    auto clipped = correct_probs.cwiseMin(max_value).cwiseMax(min_value) ; 
    auto loged = clipped.array().log() * -1 ;
    auto average_loss = loged.mean() ;
     
    return average_loss ;
    
}

Eigen::VectorXd CrossEntropyError::select_correct_probs(const Eigen::MatrixXd& prediction,const Eigen::MatrixXd& actual) {
    const int N = prediction.rows();
    const int C = prediction.cols();

    Eigen::VectorXd result(N);

    // Case A: actual is a row vector of class indices → shape (1 x N)
    if (actual.rows() == 1 && actual.cols() == N) {

        for (int i = 0; i < N; ++i) {
            int col = static_cast<int>(actual(0, i));
            result(i) = prediction(i, col);
        }
    }

    // Case B: actual is one-hot → shape (N x C)
    else if (actual.rows() == N && actual.cols() == C) {

        // Vectorized "np.sum(prediction * actual, axis=1)"
        result = (prediction.array() * actual.array()).rowwise().sum();
    }

    else {
        std::cout << "Shape mismatch: actual must be 1xN or NxC, but actual shape is :" << actual.rows() << "x" << actual.cols() << std::endl ;
        throw std::runtime_error("Shape mismatch: actual must be 1xN or NxC.");
    }

    return result ;
}