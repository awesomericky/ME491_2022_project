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
    obDim_ = 34 - 1 + 6 + 3;
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
      server_->launchServer();
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

    double goalReward = 10. - (goal_position - gc_.head(3)).norm();

    rewards_.record("torque", aliengo_->getGeneralizedForce().squaredNorm());
    rewards_.record("goal", goalReward);

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

    // compute relative target goal position
    for (int i=0; i<obstacles_.size()-1; i++) {
      if (obstacle_x_pos[i+1] > gc_[0] && gc_[0] >= obstacle_x_pos[i]) {
        closestObsRelPos = rot.e().transpose() * (obstacles_[i+1]->getPosition() - gc_.head(3));
        goalObsRelPos = rot.e().transpose() * (goal_position - gc_.head(3));
      }
    }

    Eigen::Vector3d obsVel ={velocity, 0, 0};
    Eigen::Vector3d obsRelVel = rot.e().transpose() * obsVel;

//    obDouble_ << gc_[2], /// body height
//        rot.e().row(2).transpose(), /// body orientation
//        gc_.tail(12), /// joint angles
//        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
//        gv_.tail(12); /// joint velocity
    obDouble_ <<
        rot.e().row(2).transpose(), /// body orientation
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12), /// joint velocity
        closestObsRelPos,
        goalObsRelPos,
        obsRelVel;
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

    if (is_success_) {
      terminalReward = 0.f;
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
      std::vector<double> height_batch = {1.0, 1.0, 1.0+0.12, 1.0-0.15, 1.0+0.3};
      std::vector<double> gap_batch = {0, 0.25+0.03, 0.75-0.02, 1.5+0.05, 2.5+0.01};
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
        for (int i=0; i<num_obstacle; i++) {
          /// Set the vertical & horizontal gap
          height = 1.0 + (i != 0) * (0.05*i) * normDist_(gen_); /// add noise
          gap = 0.25*i*(i+1)/2 + (i != 0) * 0.05 * normDist_(gen_); /// add noise

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
  double terminalRewardCoeff_ = -10.;
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


  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

