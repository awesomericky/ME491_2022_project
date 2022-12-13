//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    aliengo_ = world_->addArticulatedSystem(resourceDir_+"/env/envs/aliengo_jump/rsc/aliengo/aliengo.urdf");
    aliengo_->setName("aliengo");
    aliengo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// Spawn obstacles
    for (int i=0; i<num_obstacle; i++) {
      /// add box
      auto obs_tmp = world_->addBox(2, 4, 0.05, 1);
      obs_tmp->setBodyType(BodyType::STATIC); /// BodyType STATIC : mass = infinite, velocity = 0 (does not move)

      /// add obstacle pointer into obstacle set
      obstacles_.push_back(obs_tmp);
    }

    /// get robot data
    gcDim_ = aliengo_->getGeneralizedCoordinateDim();
    gvDim_ = aliengo_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    gc_nominal_.setZero(gcDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of aliengo
    gc_nominal_ << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8;
    gc_init_ = gc_nominal_;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    aliengo_->setPdGains(jointPgain, jointDgain);
    aliengo_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 34 - 1 + 6 + 3 + 2 + 1 + 1 + 1;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(aliengo_->getBodyIdx("FR_calf"));
    footIndices_.insert(aliengo_->getBodyIdx("FL_calf"));
    footIndices_.insert(aliengo_->getBodyIdx("RR_calf"));
    footIndices_.insert(aliengo_->getBodyIdx("RL_calf"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer(8080);
      server_->focusOn(aliengo_);
      /// add Visual sphere
      goal_sphere = server_->addVisualSphere("goal_obj_", 0.3, 1, 0, 0, 0.7);
    }
  }

  void init() final { }

  void reset(bool test=false) final {
    obstacleReset(test);
    gc_init_ = update_gc_init(gc_nominal_);
    aliengo_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    aliengo_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      obstacleUpdate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    Eigen::Vector3d goalHeading = goalObsRelPos / goalObsRelPos.head<2>().norm();
    goalHeading[2] = 0.;
    Eigen::Vector2d heading; heading << 1., 0.;
//    double projGoalDist = std::min(goalObsRelPos.norm(), 3.);

    double goalReward = 5. - goalObsRelPos.norm(); //////////////
//    if (goalObsRelPos.norm() < 1.) {
//      goalReward += 100 - 100 * (aliengo_->getBasePosition().e() - goal_position).head(2).norm();
//    }  /// Try2

    double headingCosTheta = heading.dot(goalHeading.head(2)) / (goalHeading.head(2).norm() * heading.norm());
//    double projVel = bodyLinearVel_.dot(closestObsRelPos) / closestObsRelPos.norm();
    double projVel = bodyLinearVel_.head(2).dot(closestObsRelPos.head(2)) / closestObsRelPos.head(2).norm();
    double orthoVel = std::sqrt(bodyLinearVel_.head(2).squaredNorm() - std::pow(projVel, 2));

    rewards_.record("torque", aliengo_->getGeneralizedForce().squaredNorm());
    /// Try1
//    rewards_.record("goal", 3 * goalReward + 10 * headingCosTheta + 10 * projVel);

    /// Try2
    if (obstacles_[num_obstacle - 2]->getPosition()[0] - 1. < gc_[0]) {
      double goalRewVal = 10 * goalReward + 10 * (projVel - orthoVel) - 100 * std::min(std::abs(gc_[1]), 0.2);
//      double goalRewVal = 10 * goalReward - 500 * std::min(std::abs(gc_[1]), 0.2);
      if (isnan(goalRewVal))
        rewards_.record("goal", 0.);
      else
        rewards_.record("goal", goalRewVal);
//      if ((aliengo_->getBasePosition().e() - goal_position).head(2).norm() < 0.2)
//        rewards_.record("goal", 20 * (projVel - orthoVel)); //3
//      else {
//        rewards_.record("goal", 20 * goalReward); //3
    } else {
      double goalRewVel = 10 * headingCosTheta + projVel - 100 * std::min(std::abs(gc_[1]), 0.2);
      if (isnan(goalRewVel))
        rewards_.record("goal", 0.);
      else
        rewards_.record("goal", goalRewVel);
//      rewards_.record("goal", 10 * headingCosTheta + 10 * projVel);
    }

//    /// Try3
//    if (obstacles_[num_obstacle - 2]->getPosition()[0] - 1. < gc_[0]) {
//      rewards_.record("goal", 50 * headingCosTheta + 20 * goalReward); //3
//    } else {
//      rewards_.record("goal", 10 * headingCosTheta + 10 * projVel);
//    }

    return rewards_.sum();
  }

  void updateObservation() {
    aliengo_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    goal_position = obstacles_.back()->getPosition();
    is_success_ = isSucessState();
    if(visualizable_)
      goal_sphere->setPosition(goal_position);

    double current_gap = 0.;
    double current_height = 0.;

    // compute relative target goal position
    for (int i=0; i<obstacles_.size()-1; i++) {
      if (obstacle_x_pos[i+1] >= gc_[0] && gc_[0] >= obstacle_x_pos[i]) {
        closestObsRelPos = rot.e().transpose() * (obstacles_[i+1]->getPosition() - gc_.head(3));
        Eigen::Vector3d goal_pos_w_height = goal_position;
        goal_pos_w_height[2] += 0.5;
        goalObsRelPos = rot.e().transpose() * (goal_pos_w_height - gc_.head(3));
        if (i == obstacles_.size() - 2) {
          current_gap = obstacles_[i+1]->getPosition()[0] - obstacles_[i]->getPosition()[0] - 2.;
        } else {
          current_gap = gap_batch[i+1];
        }

        current_height = height_batch[i+1] - gc_[2];
        targetIdx = i+1;
        break;
      }
    }

    if (obstacle_x_pos[0] > gc_[0]) {
      closestObsRelPos = rot.e().transpose() * (obstacles_[1]->getPosition() - gc_.head(3));
      Eigen::Vector3d goal_pos_w_height = goal_position;
      goal_pos_w_height[2] += 0.5;
      goalObsRelPos = rot.e().transpose() * (goal_pos_w_height - gc_.head(3));
      current_gap = 0.;
      current_height = height_batch[1] - gc_[2];
      targetIdx = 1;
    }

    if (gc_[0] >= obstacle_x_pos[num_obstacle - 1]) {
      closestObsRelPos = rot.e().transpose() * (obstacles_[num_obstacle - 1]->getPosition() - gc_.head(3));
      Eigen::Vector3d goal_pos_w_height = goal_position;
      goal_pos_w_height[2] += 0.5;
      goalObsRelPos = rot.e().transpose() * (goal_pos_w_height - gc_.head(3));
      current_gap = 0.;
      current_height = height_batch[num_obstacle - 1] - gc_[2];
      targetIdx = num_obstacle - 1;
    }

    Eigen::Vector3d obsVel ={velocity, 0, 0};
    Eigen::Vector3d obsRelVel = rot.e().transpose() * obsVel;

//    obDouble_ << gc_[2], /// body height
//        rot.e().row(2).transpose(), /// body orientation
//        gc_.tail(12), /// joint angles
//        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
//        gv_.tail(12); /// joint velocity
    obDouble_ << gc_[1], gc_[2],
        rot.e().row(2).transpose(), /// body orientation
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12), /// joint velocity
        closestObsRelPos,
        goalObsRelPos,
        obsRelVel,
        current_gap,
        current_height,
        (aliengo_->getBasePosition().e() - goal_position).head(2).norm();
  }

  Eigen::VectorXd update_gc_init (Eigen::VectorXd gc_nominal) {
    Eigen::VectorXd gc_init = gc_nominal;
    gc_init[2] += obstacle_heights.front();
    return gc_init;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: aliengo_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    for (auto & obstacle : obstacles_) {
      if (obstacle->getPosition()[0] - 1. < gc_[0] && gc_[0] < obstacle->getPosition()[0] + 1.)
        if (gc_[2] <= obstacle->getPosition()[2])
          return true;
    }

    if (obstacles_.back()->getPosition()[0] + 1. <= gc_[0])
      return true;

    if (is_success_) {
      terminalReward = 10.; // 100 /////////////////////
      return true;
    }

    terminalReward = 0.f;
    return false;
  }

  bool isSucessState() {
    double offset = 0.05; //Verifying success state
    double distance = (aliengo_->getBasePosition().e() - goal_position).head(2).norm(); // distance between robot base pos, goal pos
    if (distance <= offset)
      return true;
    return false;
  }

  void curriculumUpdate() { };

  void obstacleUpdate() {
    double gap_offset = 0.6;
    if (std::abs(obstacle_x_pos.back() - obstacles_.back()->getPosition()(0)) > gap_offset)
    {
      velocity *= -1;
      while (std::abs(obstacle_x_pos.back() - obstacles_.back()->getPosition()(0)) > gap_offset) {
        Eigen::Vector3d position_offset={velocity, 0, 0};
        position_offset += obstacles_.back()->getPosition();
        obstacles_.back()->setPosition(position_offset);
      }
    }

    Eigen::Vector3d position_offset={velocity, 0, 0};
    position_offset += obstacles_.back()->getPosition();
    obstacles_.back()->setPosition(position_offset);

    /// For angular perturbation (do not use)
//    double angular_velocity = M_PI/18 * simulation_dt_;
//    double angular_gap_offset = M_PI/36; // 5 degree
//
//    if (std::abs(obstacles_.back()->getOrientation().e().row(0)(2)) > std::abs(sin(angular_gap_offset)))
//      angular_velocity *= -1;
//
//    Eigen::Matrix3d rotation_offset;
//    rotation_offset << cos(angular_velocity), 0, sin(angular_velocity),
//        0, 1, 0,
//        -sin(angular_velocity), 0, cos(angular_velocity);
//    rotation_offset = rotation_offset*obstacles_.back()->getOrientation().e();
//    obstacles_.back()->setOrientation(rotation_offset);
  }

  void obstacleReset(bool test=false) {

    obstacle_heights.clear();
    obstacle_x_pos.clear();
    double height;
    double random_ = 0.;
    double gap;
    /// For test
    if (test) {
      height_batch = {1.0, 1.0, 1.0+0.12, 1.0-0.15, 1.0+0.3};
      gap_batch = {0, 0.25+0.03, 0.75-0.02, 1.5+0.05, 2.5+0.01};
      for (int i = 0; i < num_obstacle; i++) {
        /// Set the vertical & horizontal gap
        height = height_batch[i]; /// add noise
        gap = gap_batch[i];

        /// Set position
        obstacles_[i]->setPosition(2 * i + gap, 0, height);

        /// add obstacle pointer into obstacle set
        obstacle_heights.push_back(height);
        obstacle_x_pos.push_back(2 * i + gap);
      }
    }

    else
    {
        height_batch.clear();
        gap_batch.clear();
        for (int i=0; i<num_obstacle; i++) {
          /// Set the vertical & horizontal gap
          height = 1.0 + (i != 0) * (0.05*i) * normDist_(gen_); /// add noise
          gap = 0.25*i*(i+1)/2 + (i != 0) * 0.05 * normDist_(gen_); /// add noise
          height_batch.push_back(height);
          gap_batch.push_back(gap);

          /// Set position
          obstacles_[i]->setPosition(2*i + gap, 0, height);

          /// add obstacle pointer into obstacle set
          obstacle_heights.push_back(height);
          obstacle_x_pos.push_back(2*i + gap);
        }
    }



  };

 private:
  double velocity = 0.6 * simulation_dt_;
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* aliengo_;
  Eigen::VectorXd gc_init_, gv_init_, gc_nominal_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
//  double terminalRewardCoeff_ = -10.;
  double terminalRewardCoeff_ = -1.;  // -1. ////////////////////////////
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  int num_obstacle = 5;
  std::vector<raisim::Box *> obstacles_;
  std::vector<double> obstacle_heights;
  std::vector<double> obstacle_x_pos;
  raisim::Visuals *goal_sphere;
  Eigen::Vector3d goal_position;
  bool is_success_ = false;

  Eigen::Vector3d closestObsRelPos, goalObsRelPos;
  std::vector<double> height_batch, gap_batch;
  int targetIdx;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

