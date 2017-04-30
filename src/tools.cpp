#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
	VectorXd rmse(4);
	rmse << 0,0,0,0;

    if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
	    std::cout << "invalid inputs for RMSE" << std::endl;
    }

	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
	    VectorXd resid = estimations[i] - ground_truth[i];
	    VectorXd sq_resid = resid.array() * resid.array();
	    rmse += sq_resid;
	}
    
	//calculate the mean
	rmse = rmse.array() / estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}
