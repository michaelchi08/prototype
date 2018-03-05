#!/bin/sh

# find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

# find . -name "*.py" -exec autopep8 --in-place {} \;

# python scripts/api.py prototype docs/api

# bash scripts/kitti_vo_data.bash

# export PYTHONPATH=/usr/local/lib/python2.7/dist-packages
export PYTHONPATH=/usr/local/lib/python3.5/dist-packages
# source /opt/ros/kinetic/setup.bash

# python3 api.py

# python -m unittest discover
# python3 -m unittest discover -b

# python3 -m unittest prototype.tests.calibration.test_preprocess
# python3 -m unittest prototype.tests.calibration.test_loader
# python3 -m unittest prototype.tests.calibration.test_dataset.GimbalDataGeneratorTest
# python3 -m unittest prototype.tests.calibration.test_calibration.GimbalCalibratorTest.test_setup_problem
# python3 -m unittest prototype.tests.calibration.test_calibration.GimbalCalibratorTest.test_reprojection_error
# python3 -m unittest prototype.tests.calibration.test_calibration.GimbalCalibratorTest.test_optimize
# python3 -m unittest prototype.tests.calibration.test_calibration.GimbalCalibratorTest.test_optimize_preprocessed
# python3 -m unittest prototype.tests.calibration.test_calibration.GimbalCalibratorDataLoaderTest

# python3 -m unittest prototype.tests.control.quadrotor.test_attitude
# python3 -m unittest prototype.tests.control.quadrotor.test_position
# python3 -m unittest prototype.tests.control.test_carrot
# python3 -m unittest prototype.tests.control.test_pid

# python3 -m unittest prototype.tests.data.test_dataset
# python3 -m unittest prototype.tests.data.test_kitti

# python3 -m unittest prototype.tests.estimation.test_kf
# python3 -m unittest prototype.tests.estimation.test_ekf
# python3 -m unittest prototype.tests.estimation.test_inverse_depth

# python3 -m unittest prototype.tests.models.test_imu
# python3 -m unittest prototype.tests.models.test_two_wheel
# python3 -m unittest prototype.tests.models.test_husky
# python3 -m unittest prototype.tests.models.test_quadrotor

# python3 -m unittest prototype.tests.msckf.test_camera_state
# python3 -m unittest prototype.tests.msckf.test_imu_state
# python3 -m unittest prototype.tests.msckf.test_feature_estimator
# python3 -m unittest prototype.tests.msckf.test_feature_estimator.FeatureEstimatorTest.test_triangulate
# python3 -m unittest prototype.tests.msckf.test_feature_estimator.FeatureEstimatorTest.test_estimate
# python3 -m unittest prototype.tests.msckf.test_feature_estimator.FeatureEstimatorTest.test_estimate2
# python3 -m unittest prototype.tests.msckf.test_msckf
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_init
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_P
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_N
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_H
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_augment_state
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_track_cam_states
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_track_residuals
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_prediction_update
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_residualize_track
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_calculate_residuals
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_estimate_feature
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_measurement_update
# python3 -m unittest prototype.tests.msckf.test_msckf.MSCKFTest.test_measurement_update2

# python3 -m unittest prototype.tests.optimization.test_residuals

# python3 -m unittest prototype.tests.utils.test_utils
# python3 -m unittest prototype.tests.utils.test_gps
# python3 -m unittest prototype.tests.utils.test_linalg
# python3 -m unittest prototype.tests.utils.test_transform
# python3 -m unittest prototype.tests.utils.quaternion.test_jpl
# python3 -m unittest prototype.tests.utils.quaternion.test_hamiltonian

# python3 -m unittest prototype.tests.vision.test_apriltag
# python3 -m unittest prototype.tests.vision.camera.test_camera
# python3 -m unittest prototype.tests.vision.camera.test_camera_model
# python3 -m unittest prototype.tests.vision.camera.test_distortion_model
# python3 -m unittest prototype.tests.vision.data.test_dataset
# python3 -m unittest prototype.tests.vision.data.test_kitti
# python3 -m unittest prototype.tests.vision.feature2d.test_fast
# python3 -m unittest prototype.tests.vision.feature2d.test_feature_container
# python3 -m unittest prototype.tests.vision.feature2d.test_feature_tracker
# python3 -m unittest prototype.tests.vision.feature2d.test_gms_matcher
# python3 -m unittest prototype.tests.vision.feature2d.test_keyframe
# python3 -m unittest prototype.tests.vision.feature2d.test_keypoint
# python3 -m unittest prototype.tests.vision.feature2d.test_lk_tracker
# python3 -m unittest prototype.tests.vision.feature2d.test_klt_tracker
# python3 -m unittest prototype.tests.vision.feature2d.test_stereo_tracker
# python3 -m unittest prototype.tests.vision.feature2d.test_orb
# python3 -m unittest prototype.tests.vision.feature2d.test_ransac
# python3 -m unittest prototype.tests.vision.test_common
# python3 -m unittest prototype.tests.vision.test_geometry
# python3 -m unittest prototype.tests.vision.test_homography

# python3 -m unittest prototype.tests.viz.test_plot_quadrotor
# python3 -m unittest prototype.tests.viz.test_plot_gimbal
# python3 -m unittest prototype.tests.viz.test_plot_chessboard
# python3 -m unittest prototype.tests.viz.test_plot_grid

# python3 prototype/viz.py

# cd docs
