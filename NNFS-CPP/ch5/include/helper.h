// In the name of Allah

#include <Eigen/Dense>
#include <random>
#include <utility>


std::pair<Eigen::MatrixXd, Eigen:: VectorXi> create_data(int samples , int classes , unsigned int seed = std::random_device{}()) {

    int total_samples = samples * classes ;
    
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(total_samples , 2);
    Eigen::VectorXi y = Eigen::VectorXi::Zero(total_samples);
    
    // Random generator for gaussian noise 
    std::mt19937 rng(seed) ;
    std::normal_distribution<double> dist(0.0, 0.2) ; 

    for(int i = 0 ; i < classes ;  ++i) {
        int start = samples * i;
        Eigen::VectorXd r = Eigen::VectorXd::LinSpaced(samples, 0.0, 1.0) ;

        Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(samples, static_cast<double>(i) * 4.0, static_cast<double>(i + 1)*4.0) ;

        // add the guassian noise individually. 

        for(int j = 0 ; j < samples ; ++j) { t(j) += dist(rng) ; }

        // Compute coordinates: x = r * sin(t*2.5), y = r * cos(t*2.5)
        // Use Eigen array() for elementwise sin/cos and multiplication
        Eigen::VectorXd xcol = (r.array() * ( (t * 2.5).array().sin() )).matrix();
        Eigen::VectorXd ycol = (r.array() * ( (t * 2.5).array().cos() )).matrix();

        // Assign into the big matrix
        X.block(start, 0, samples, 1) = xcol;
        X.block(start, 1, samples, 1) = ycol;

        // Labels
        y.segment(start, samples).setConstant(i);


    }

    return {X, y} ;
};