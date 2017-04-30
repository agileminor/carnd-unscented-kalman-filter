#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <math.h>
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = .6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  n_x_ = 5;
  is_initialized_ = false;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;
  Xsig_ = MatrixXd(n_x_, 2 * n_x_ + 1);
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  previous_timestamp_ = 0;
  //create sigma point matrix
}

UKF::~UKF() {}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
	//cout << "start prediction" << endl;

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);
  

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  VectorXd weights = VectorXd(2*n_aug_+1);
  //set weights
  double w1 = lambda_ / (lambda_ + n_aug_);
  double w2 = 1 / (2 * (lambda_ + n_aug_));
  
  weights(0) = w1;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    weights(i) = w2;
  }


 
  //create augmented mean state
  //cout << "make aug mean state" << endl;
  x_aug.head(5) = x_;
  x_aug(5) = 0.0;
  x_aug(6) = 0.0;
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;
  //create augmented covariance matrix
  //create square root matrix
  MatrixXd rt = P_aug.llt().matrixL();
  //create augmented sigma points
    Xsig_aug_.col(0) = x_aug;
  float prefix_x = sqrt(lambda_ + n_aug_);
  for (int i=0;i < n_aug_; i++) {
      Xsig_aug_.col(i+1) = x_aug + prefix_x * rt.col(i);
      Xsig_aug_.col(i + n_aug_ +1) = x_aug - prefix_x * rt.col(i);
  }


  VectorXd x_k = VectorXd(n_x_);
  VectorXd vec1 = VectorXd(n_x_);
  VectorXd vec2 = VectorXd(n_x_);
  float delta_t_2 = 0.5 * delta_t * delta_t;
  //cout << "predict sigma points" << endl;
  //predict sigma points
    for (int i=0; i < 2 * n_aug_ + 1; i++) {
      float px = Xsig_aug_(0, i);
      float py = Xsig_aug_(1, i);
      float v = Xsig_aug_(2, i);
      float yaw = Xsig_aug_(3, i);
      float yaw_dot = Xsig_aug_(4, i);
      float va = Xsig_aug_(5, i);
      float vyaw = Xsig_aug_(6, i);
      x_k << px,
             py,
             v,
             yaw,
             yaw_dot;
      vec2 << delta_t_2 * cos(yaw) * va, 
              delta_t_2 *  sin(yaw) * va,
              delta_t * va,
              delta_t_2 * vyaw,
              delta_t * vyaw;
      // changed 3rd entry to match lecture notes, from 0
        if (fabs(yaw_dot) < 0.00001) {
            vec1 << v * cos(yaw) * delta_t,
                    v * sin(yaw) * delta_t,
                    0,
                    yaw_dot * delta_t,
                    0;
        } else {
            vec1 << v/ yaw_dot * (sin(yaw + yaw_dot * delta_t) - sin(yaw)),
                    v / yaw_dot * (-cos(yaw + yaw_dot * delta_t) + cos(yaw)),
                    0,
                    yaw_dot * delta_t,
                    0;
        }
        Xsig_pred_.col(i) = x_k + vec1 + vec2;      
  }
  

  //predict state mean
  //cout << "predict state mean" << endl;
  for (int i =0;i < n_x_;i++) {
      float temp = 0.0;
      for (int j=0;j < 2* n_aug_ + 1; j++) {
          if (j==0) {
              temp = temp + w1 * Xsig_pred_(i, j);
          } else {
              temp = temp + w2 * Xsig_pred_(i, j);
          }
      }
      x_(i) = temp;
  }
  //predict state covariance matrix
  //cout << "predict state covariance matrix" << endl;
    P_.fill(0.0);
    for (int i=0; i < 2 * n_aug_ + 1; i++) {
        VectorXd temp_x = Xsig_pred_.col(i) - x_;
            //angle normalization
    while (temp_x(3)> M_PI) temp_x(3)-=2.*M_PI;
    while (temp_x(3)<-M_PI) temp_x(3)+=2.*M_PI;
        P_ = P_ + weights(i) * temp_x * temp_x.transpose();
    }
    //cout << "end prediction" << endl;
}
/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const VectorXd &z) {
  /**

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //cout << "update radar" << endl;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;


  //set vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);
   double weight_0 = lambda_/(lambda_+n_aug_);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  
    double weight = 0.5/(n_aug_+lambda_);
    weights(i) = weight;
  }

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  z_pred.fill(0.0);
  //transform sigma points into measurement space
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
      double px = Xsig_pred_(0, i);
      double py = Xsig_pred_(1, i);
      double v = Xsig_pred_(2, i);
      double phi = Xsig_pred_(3, i);
      double phi_dot = Xsig_pred_(4, i);
      double temp1 = sqrt(pow(px, 2) + pow(py, 2));
      Zsig(0, i) = temp1;
      Zsig(1, i) = atan2(py, px);
      Zsig(2, i) = (px * cos(phi) * v + py * sin(phi) * v) / temp1;
      z_pred = z_pred + weights(i) * Zsig.col(i);
  }
  //calculate mean predicted measurement

  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd(n_z, n_z);
  R << pow(std_radr_,2), 0, 0,
       0, pow(std_radphi_,2), 0,
       0, 0, pow(std_radrd_,2);
    S.fill(0.0);
    for (int i=0; i < 2 * n_aug_ + 1; i++) {
        VectorXd temp_z = Zsig.col(i) - z_pred;
        while (temp_z(1)> M_PI) temp_z(1)-=2.*M_PI;
        while (temp_z(1)<-M_PI) temp_z(1)+=2.*M_PI;

        S = S + weights(i) * temp_z * temp_z.transpose();
    }
    S = S + R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);


  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i=0;i < 2 * n_aug_ + 1; i++) {
          //angle normalization
    VectorXd z_diff = Zsig.col(i) -z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
      Tc = Tc + weights(i) * (Xsig_pred_.col(i) - x_) * z_diff.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //update state mean and covariance matrix
      VectorXd z_diff = z -z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    //cout << "done update radar, NIS=" << NIS_radar_ << endl;

}
/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const VectorXd &z) {
	//cout << "update lidar" << endl;
	MatrixXd H_ = MatrixXd(2,5);
	H_ << 1, 0, 0, 0, 0,
	      0, 1, 0, 0, 0;
  MatrixXd R_ = MatrixXd(2, 2);
  R_ << pow(std_laspx_,2), 0,
       0, pow(std_laspy_,2);
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
	NIS_laser_ = y.transpose() * Si * y;
	//cout << "done update lidar, NIS lidar= " << NIS_laser_ << endl;
}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
	if (!is_initialized_) {
		//cout << "initializing" << endl;
		P_ << 1, 0, 0, 0, 0,
	      0, 1, 0, 0, 0,
	      0, 0, 1, 0, 0,
	      0, 0, 0, 1, 0,
	      0, 0, 0, 0, 1;
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			//cout << "initialize radar" << endl;
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
	double rho = meas_package.raw_measurements_[0];
	double phi = meas_package.raw_measurements_[1];

	double px = rho * cos(phi);
	double py = rho * sin(phi);

	x_ << px, py, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
	    //cout << "initialize laser" << endl;
	x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    previous_timestamp_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
	double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = meas_package.timestamp_;
	Prediction(delta_t);
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package.raw_measurements_);
  } else {
    // Laser updates
    UpdateLidar(meas_package.raw_measurements_);
  }
// print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}
