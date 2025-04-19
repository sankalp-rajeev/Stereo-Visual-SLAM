#!/usr/bin/env python

import carla
import numpy as np
import cv2
import os
import time
import threading
import queue
import json
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D  # Needed even if not used directly

# Added for 3D point cloud/mapping
import open3d as o3d

# =======================================
# Simple 9D Kalman Filter for [px, py, pz, vx, vy, vz, rx, ry, rz]
# where (rx, ry, rz) could be Euler angles (roll, pitch, yaw).
# This is a minimal placeholder for orientation + position fusion.
# For production use, a quaternion-based or 15D state (including gyro biases) is typical.
# =======================================
class KalmanFilter9D:
    def __init__(self, init_state):
        """
        init_state = [px, py, pz, vx, vy, vz, roll, pitch, yaw]
        """
        self.x = np.array(init_state, dtype=np.float32).reshape(9, 1)
        # Covariance
        self.P = np.eye(9) * 1.0
        # Process noise
        self.Q = np.eye(9) * 0.01
        # Measurement noise: here we separate position + orientation
        # Let's say we get position from vision and orientation from gyroscope integration
        self.R_pos = np.eye(3) * 0.05
        self.R_ori = np.eye(3) * 0.05
        self.I = np.eye(9)

    def predict(self, dt, acc, gyro):
        """
        Predict step using constant velocity + constant rotation rate model.
        acc = (ax, ay, az) in m/s^2
        gyro = (wx, wy, wz) in rad/s (roll rate, pitch rate, yaw rate)
        """
        # State x = [px, py, pz, vx, vy, vz, roll, pitch, yaw]

        # 1) Build the system matrix A
        A = np.eye(9, dtype=np.float32)
        # position update: p(k+1) = p(k) + v(k)*dt
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt
        # orientation update: euler(k+1) = euler(k) + gyro*dt
        # (We keep them the same in state, but we incorporate them via B control matrix)

        # 2) Control matrix B to incorporate acceleration into velocity + gyro into orientation
        B = np.zeros((9, 6), dtype=np.float32)  # We have 6 controls: (ax, ay, az, wx, wy, wz)
        # velocity update from acc
        B[3, 0] = dt
        B[4, 1] = dt
        B[5, 2] = dt
        # orientation update from gyro
        B[6, 3] = dt
        B[7, 4] = dt
        B[8, 5] = dt

        u = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]], dtype=np.float32).reshape(6, 1)

        # Predict
        self.x = A @ self.x + B @ u
        self.P = A @ self.P @ A.T + self.Q

    def update_position(self, measured_pos):
        """
        Measured position is [px, py, pz].
        We'll only update that part of state,
        ignoring orientation here.
        """
        H = np.zeros((3, 9), dtype=np.float32)
        # Position is in x[0:3]
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        z = np.array(measured_pos, dtype=np.float32).reshape(3, 1)
        y = z - H @ self.x  # innovation
        S = H @ self.P @ H.T + self.R_pos
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

    def update_orientation(self, measured_euler):
        """
        measured_euler = [roll, pitch, yaw]
        """
        H = np.zeros((3, 9), dtype=np.float32)
        # orientation is in x[6:9]
        H[0, 6] = 1
        H[1, 7] = 1
        H[2, 8] = 1

        z = np.array(measured_euler, dtype=np.float32).reshape(3, 1)
        y = z - H @ self.x  # innovation
        S = H @ self.P @ H.T + self.R_ori
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

# =========================================================
# VisualSLAM class that does stereo matching each frame,
# extracts 3D points from matched keypoints in the left image,
# solves a PnP to get the transformation from the last frame.
# If loop closure is detected, we just print a note.
# Additionally, it accumulates a global 3D map from keyframes.
# =========================================================
class StereoVisualSLAM:
    def __init__(self, f, cx, cy, baseline):
        """
        f       = focal length in pixels (assumed same for left/right).
        cx, cy  = principal point coords in the left camera.
        baseline = distance between left and right camera in meters.
        """
        self.f = f
        self.cx = cx
        self.cy = cy
        self.baseline = baseline

        # Attempt SIFT first, fallback to ORB
        try:
            self.detector = cv2.SIFT_create()
            self.feature_type = 'SIFT'
        except:
            self.detector = cv2.ORB_create(3000)
            self.feature_type = 'ORB'

        # FLANN matcher
        if self.feature_type == 'SIFT':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            # ORB case
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)

        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Create stereo matcher
        # For a real system, you would rectify the images and tune these parameters carefully
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=32,
            disp12MaxDiff=1,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2
        )

        # Save the last frame’s data
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None  # 3D points from the last frame
        self.pose = np.eye(4, dtype=np.float32)
        self.trajectory = []

        # For loop closure
        self.keyframe_descriptors = []
        self.keyframe_poses = []

        # Global map accumulation: list of Nx3 sets of points
        self.global_points = []

    def compute_stereo_depth(self, left_img_gray, right_img_gray):
        """
        Compute disparity via SGBM, then convert to depth in meters:
          depth = f*baseline / disparity
        """
        disparity = self.stereo_matcher.compute(left_img_gray, right_img_gray).astype(np.float32) / 16.0
        # Avoid divide by zero
        disparity[disparity <= 0.1] = 0.1

        depth_map = self.f * self.baseline / disparity  # in meters
        return depth_map

    def detect_loop_closure(self, des):
        # Very naive: if we have at least 5 keyframes, compare descriptors with older frames
        if len(self.keyframe_descriptors) < 5 or des is None:
            return False, None
        search_range = len(self.keyframe_descriptors) - 10
        if search_range <= 0:
            return False, None

        best_idx = -1
        best_match_count = 0
        for i in range(search_range):
            if self.keyframe_descriptors[i] is None:
                continue
            matches = self.matcher.match(des, self.keyframe_descriptors[i])
            matches = sorted(matches, key=lambda x: x.distance)
            good = [m for m in matches if m.distance < 50]
            if len(good) > 30 and len(good) > best_match_count:
                best_match_count = len(good)
                best_idx = i
        if best_idx >= 0:
            return True, self.keyframe_poses[best_idx]
        return False, None

    def _extract_3d_points(self, keypoints, depth_map):
        """
        Convert a set of 2D keypoints in the left image into 3D points
        using the stereo-derived depth_map.
        X = (u - cx)*Z / f
        Y = (v - cy)*Z / f
        Z = depth_map[v, u]
        """
        pts_3d = []
        for kp in keypoints:
            u, v = int(kp.pt[0]), int(kp.pt[1])
            if u < 0 or v < 0 or u >= depth_map.shape[1] or v >= depth_map.shape[0]:
                pts_3d.append([0,0,0])
                continue
            Z = depth_map[v, u]  # in meters
            if Z <= 0.0 or Z > 200:
                pts_3d.append([0,0,0])
                continue
            X = (u - self.cx)*Z / self.f
            Y = (v - self.cy)*Z / self.f
            pts_3d.append([X, Y, Z])
        return np.array(pts_3d, dtype=np.float32)

    def process_stereo_frame(self, left_img, right_img):
        """
        1) Convert to gray
        2) Compute disparity -> depth
        3) Detect features in left_img, compute 3D points from (u,v, depth[u,v])
        4) Match to previous frame’s descriptors
        5) Solve PnP
        6) Check loop closure
        7) Possibly update keyframe
        """
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        depth_map = self.compute_stereo_depth(left_gray, right_gray)

        keypoints, descriptors = self.detector.detectAndCompute(left_gray, None)
        if self.prev_keypoints is None:
            # First frame
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_points_3d = self._extract_3d_points(keypoints, depth_map)
            self.keyframe_descriptors.append(descriptors)
            self.keyframe_poses.append(self.pose.copy())
            # Store trajectory
            self.trajectory.append((self.pose[0, 3], self.pose[1, 3], self.pose[2, 3]))
            return {'success': True, 'pose': self.pose}

        # Match to previous descriptors
        if descriptors is None or self.prev_descriptors is None:
            return {'success': False}

        # KNN match
        matches_knn = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        pts_prev = []
        pts_curr = []
        pts_3d = []
        good_matches = []
        ratio_thresh = 0.75

        for m, n in matches_knn:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                # prev frame keypoint
                kp_prev = self.prev_keypoints[m.queryIdx].pt
                # current frame keypoint
                kp_curr = keypoints[m.trainIdx].pt
                # 3D from the prev frame
                X = self.prev_points_3d[m.queryIdx]
                # Only use valid 3D
                if X[2] > 0.01 and X[2] < 100:  # 0.01m < Z < 100m
                    pts_prev.append(kp_prev)
                    pts_curr.append(kp_curr)
                    pts_3d.append(X)

        if len(pts_3d) < 8:
            return {'success': False}

        obj_points = np.array(pts_3d, dtype=np.float32)
        img_points = np.array(pts_curr, dtype=np.float32)

        # Intrinsic matrix
        K = np.array([[self.f, 0, self.cx],
                      [0, self.f, self.cy],
                      [0,    0,    1]], dtype=np.float32)

        # Solve PnP from last frame's 3D to this frame's 2D
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, img_points, K, None)
        if not retval or inliers is None or len(inliers) < 8:
            return {'success': False}

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

        # The transformation we get is "prev -> current"
        T_prev_current = np.eye(4, dtype=np.float32)
        T_prev_current[:3, :3] = R
        T_prev_current[:3, 3] = t

        # So absolute pose is pose * T_prev_current
        self.pose = self.pose @ T_prev_current

        # Check loop closure
        loop_detected, old_pose = self.detect_loop_closure(descriptors)
        if loop_detected:
            print("Loop closure detected! A global optimization step would happen here.")

        # Possibly update the keyframe if motion is large enough
        if np.linalg.norm(t) > 0.05:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_points_3d = self._extract_3d_points(keypoints, depth_map)

            self.keyframe_descriptors.append(descriptors)
            self.keyframe_poses.append(self.pose.copy())

            # -- Add newly extracted 3D points to global map --
            # Convert them to world coords using the new pose
            pts_cam = self.prev_points_3d
            ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
            pts_cam_hom = np.hstack((pts_cam, ones))  # Nx4
            pts_global_hom = (self.pose @ pts_cam_hom.T).T  # Nx4
            pts_global = pts_global_hom[:, :3]  # Nx3
            self.global_points.append(pts_global)

        # Append trajectory
        self.trajectory.append((self.pose[0, 3], self.pose[1, 3], self.pose[2, 3]))

        return {'success': True, 'pose': self.pose}

    def visualize_trajectory(self):
        """
        Plots the 2D trajectory (X-Z).
        """
        traj = np.array(self.trajectory, dtype=np.float32)
        if len(traj) < 2:
            return
        plt.figure()
        plt.plot(traj[:, 0], traj[:, 2], '-o', markersize=3)
        plt.title("2D Trajectory (X-Z)")
        plt.xlabel("X (m)")
        plt.ylabel("Z (m)")
        plt.grid(True)
        plt.savefig("vslam_final_results/trajectory_2d.png", dpi=300)
        plt.show(block=False)
        plt.pause(0.001)
        plt.close()

    def visualize_trajectory_3d(self):
        """
        Plots the 3D trajectory using matplotlib's 3D axis.
        """
        valid_points = []
        for pt in self.trajectory:
            try:
                pt_float = [float(pt[0]), float(pt[1]), float(pt[2])]
                valid_points.append(pt_float)
            except (IndexError, TypeError, ValueError):
                print("Skipping invalid trajectory point:", pt)

        if len(valid_points) < 2:
            print("Not enough valid trajectory points to visualize in 3D.")
            return

        traj = np.array(valid_points, dtype=np.float32)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-o', markersize=3)
        ax.set_title("3D Trajectory")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        plt.savefig("vslam_final_results/trajectory_3d.png", dpi=300)
        plt.show()

    def visualize_map(self):
        """
        Builds an Open3D point cloud from all global map points, optionally
        draws the trajectory in 3D, and performs optional downsampling / outlier removal.
        """
        if not self.global_points:
            print("No map points to visualize.")
            return

        # ---------- Build one big Nx3 array of map points ----------
        all_pts = np.vstack(self.global_points).astype(np.float32)

        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)

        # Optional color by height (Z)
        z_vals = all_pts[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        z_norm = (z_vals - z_min) / (z_max - z_min + 1e-6)
        cmap = plt.get_cmap('viridis')(z_norm)
        pcd.colors = o3d.utility.Vector3dVector(cmap[:, :3])
        
        # ---------- Optional: Voxel Downsample for speed ----------
        # Uncomment if the point cloud is huge:
        # pcd = pcd.voxel_down_sample(voxel_size=0.1)

        # ---------- Optional: Remove outliers (statistical outlier removal) ----------
        # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # ---------- Optionally, add the entire camera trajectory as a line set ----------
        traj = np.array(self.trajectory, dtype=np.float32)
        # We expect Nx3 shape
        if traj.shape[0] >= 2 and traj.shape[1] == 3:
            trajectory_lines = o3d.geometry.LineSet()
            trajectory_lines.points = o3d.utility.Vector3dVector(traj)
            # Make segments between consecutive trajectory points
            lines = [[i, i+1] for i in range(len(traj)-1)]
            trajectory_lines.lines = o3d.utility.Vector2iVector(lines)
            # color them red for visibility
            trajectory_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])

            # Show them in the same scene
            o3d.visualization.draw_geometries(
                [pcd, trajectory_lines],
                window_name="Global SLAM Map w/ Trajectory",
                width=800,
                height=600
            )
        else:
            # No lines, just the point cloud
            o3d.visualization.draw_geometries(
                [pcd],
                window_name="Global SLAM Map",
                width=800,
                height=600
            )

        o3d.io.write_point_cloud("vslam_final_results/slam_map.ply", pcd)

    def rotationMatrixToEulerAngles(self, R):
        """
        Convert 3x3 rotation matrix to (roll, pitch, yaw), in radians.
        See: https://learnopencv.com/rotation-matrix-to-euler-angles/
        """
        sy = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll  = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw   = np.arctan2(R[1,0], R[0,0])
        else:
            roll  = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw   = 0
        return (roll, pitch, yaw)

# =========================================================
# Main CARLA pipeline
# =========================================================
class CarlaStereoVSLAMPipeline:
    def __init__(self, host='localhost', port=2000, width=800, height=600, fov=90,
                 baseline=0.54, freq=20, show_display=True):
        """
        baseline is the distance between left and right cameras in meters (e.g., 0.54).
        freq is how many frames per second we attempt to process.
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.width = width
        self.height = height
        self.fov = float(fov)
        self.freq = freq
        self.show_display = show_display

        # Approx focal length in pixels from fov
        # f = W / (2 * tan(fov/2))
        self.f = width / (2.0 * np.tan(np.radians(self.fov/2.0)))
        self.cx = width/2.0
        self.cy = height/2.0

        self.baseline = baseline

        # Queues for sensor data
        self.left_queue = queue.Queue()
        self.right_queue = queue.Queue()
        self.imu_queue = queue.Queue()

        self.actor_list = []

        # Create the stereo SLAM instance
        self.vslam = StereoVisualSLAM(
            f=self.f,
            cx=self.cx,
            cy=self.cy,
            baseline=self.baseline
        )

        # Minimal 9D Kalman filter: [px, py, pz, vx, vy, vz, roll, pitch, yaw]
        self.kf = KalmanFilter9D([0,0,0, 0,0,0, 0,0,0])
        self.last_kf_time = None

    def spawn_vehicle(self, vehicle_type='vehicle.tesla.model3'):
        bp = self.bp_lib.find(vehicle_type)
        spawn_point = np.random.choice(self.spawn_points)
        vehicle = self.world.try_spawn_actor(bp, spawn_point)
        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle.")
        self.actor_list.append(vehicle)
        return vehicle

    def left_cam_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3]
        ts = image.timestamp
        # Store in queue
        self.left_queue.put((bgr, ts))

    def right_cam_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3]
        ts = image.timestamp
        self.right_queue.put((bgr, ts))

    def imu_callback(self, imu):
        acc = (imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z)
        gyro = (imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z)
        compass = imu.compass
        ts = imu.timestamp
        self.imu_queue.put({'acc': acc, 'gyro': gyro, 'compass': compass, 'ts': ts})

    def attach_stereo_cameras(self, vehicle):
        """
        Attach left camera at y = -baseline/2, right camera at y = +baseline/2
        so that total separation is 'baseline'.
        """
        # Left camera
        left_bp = self.bp_lib.find('sensor.camera.rgb')
        left_bp.set_attribute('image_size_x', str(self.width))
        left_bp.set_attribute('image_size_y', str(self.height))
        left_bp.set_attribute('fov', str(self.fov))
        trans_left = carla.Transform(carla.Location(x=1.6, y=-self.baseline/2, z=2.0))
        left_cam = self.world.spawn_actor(left_bp, trans_left, attach_to=vehicle)
        left_cam.listen(self.left_cam_callback)
        self.actor_list.append(left_cam)
        print("Left camera attached.")

        # Right camera
        right_bp = self.bp_lib.find('sensor.camera.rgb')
        right_bp.set_attribute('image_size_x', str(self.width))
        right_bp.set_attribute('image_size_y', str(self.height))
        right_bp.set_attribute('fov', str(self.fov))
        trans_right = carla.Transform(carla.Location(x=1.6, y=self.baseline/2, z=2.0))
        right_cam = self.world.spawn_actor(right_bp, trans_right, attach_to=vehicle)
        right_cam.listen(self.right_cam_callback)
        self.actor_list.append(right_cam)
        print("Right camera attached.")

    def attach_imu(self, vehicle):
        imu_bp = self.bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.01')  # 100 Hz
        trans_imu = carla.Transform(carla.Location(x=0, y=0, z=1.0))
        imu = self.world.spawn_actor(imu_bp, trans_imu, attach_to=vehicle)
        imu.listen(self.imu_callback)
        self.actor_list.append(imu)
        print("IMU attached.")

    def set_weather(self, preset='clear'):
        presets = {
            'clear': carla.WeatherParameters.ClearNoon,
            'cloudy': carla.WeatherParameters.CloudyNoon,
            'rain': carla.WeatherParameters.MidRainyNoon,
            'sunset': carla.WeatherParameters.SoftRainSunset
        }
        if preset in presets:
            self.world.set_weather(presets[preset])
        print(f"Weather set to {preset}")

    def run_slam(self, duration=30.0):
        """
        Main loop that runs for `duration` seconds, grabbing
        stereo frames + IMU, running SLAM, and updating our 9D Kalman filter.
        """
        start_time = time.time()
        last_frame_time = time.time()
        predicted_positions = []
        corrected_positions = []


        while time.time() - start_time < duration:
            current_time = time.time()
            dt = current_time - (self.last_kf_time if self.last_kf_time else current_time)
            self.last_kf_time = current_time

            # Try to get the most recent left & right images
            left_img, right_img = None, None
            left_ts, right_ts = None, None

            while not self.left_queue.empty():
                left_img, left_ts = self.left_queue.get()
            while not self.right_queue.empty():
                right_img, right_ts = self.right_queue.get()

            # IMU: accumulate the last reading
            imu_data = None
            while not self.imu_queue.empty():
                imu_data = self.imu_queue.get()

            if left_img is None or right_img is None:
                # No new stereo pair yet
                time.sleep(0.001)
                continue

            # Attempt to process
            result = self.vslam.process_stereo_frame(left_img, right_img)

            if result['success']:
                # If we also have IMU, do a predict + update
                if imu_data:
                    ax, ay, az = imu_data['acc']
                    gx, gy, gz = imu_data['gyro']
                    # Convert raw gyro from deg/s to rad/s if needed
                    # (Check CARLA docs for exact units.)
                    self.kf.predict(dt, (ax, ay, az), (gx, gy, gz))
                    predicted_positions.append(self.kf.x[:3].flatten().copy())
                    # We have a new vision-based pose in self.vslam.pose
                    # Extract position + Euler angles
                    pose = self.vslam.pose
                    px, py, pz = pose[0, 3], pose[1, 3], pose[2, 3]
                    # Extract euler from R
                    R = pose[:3, :3]
                    roll, pitch, yaw = self.rotationMatrixToEulerAngles(R)
                    # Update Kalman with vision-based position + orientation
                    self.kf.update_position([px, py, pz])
                    corrected_positions.append(self.kf.x[:3].flatten().copy())
                    self.kf.update_orientation([roll, pitch, yaw])

                # If display is on, show a debug window
                if self.show_display:
                    disp_img = left_img.copy()
                    cv2.putText(disp_img,
                                f"Pose: x={self.vslam.pose[0,3]:.2f}, y={self.vslam.pose[1,3]:.2f}, z={self.vslam.pose[2,3]:.2f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0,255,0),
                                2)
                    cv2.imshow("Left Camera (SLAM debug)", disp_img)
                    if cv2.waitKey(1) == 27:
                        break

            # Limit the loop to ~self.freq
            if (time.time() - last_frame_time) < (1.0 / self.freq):
                time.sleep(0.001)
                continue
            last_frame_time = time.time()
        # Save logged data
        np.save("vslam_final_results/predicted_positions.npy", np.array(predicted_positions))
        np.save("vslam_final_results/corrected_positions.npy", np.array(corrected_positions))

        # Done capturing frames
        cv2.destroyAllWindows()

        # Visualize final trajectory
        self.vslam.visualize_trajectory()
        # 3D trajectory in matplotlib
        self.vslam.visualize_trajectory_3d()
        # Show the 3D map in Open3D
        self.vslam.visualize_map()

    def rotationMatrixToEulerAngles(self, R):
        """
        Convert 3x3 rotation matrix to (roll, pitch, yaw), in radians.
        See: https://learnopencv.com/rotation-matrix-to-euler-angles/
        """
        sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
        singular = sy < 1e-6
        if not singular:
            roll  = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw   = np.arctan2(R[1,0], R[0,0])
        else:
            roll  = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw   = 0
        return (roll, pitch, yaw)

    def cleanup(self):
        for a in self.actor_list:
            if a is not None:
                a.destroy()
        print("All actors destroyed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--width", default=800, type=int)
    parser.add_argument("--height", default=600, type=int)
    parser.add_argument("--fov", default=90, type=float)
    parser.add_argument("--baseline", default=0.54, type=float, help="Stereo baseline in meters")
    parser.add_argument("--duration", default=30, type=int, help="Duration of the SLAM run in seconds")
    parser.add_argument("--weather", default="clear", choices=["clear","cloudy","rain","sunset"])
    parser.add_argument("--no-display", action="store_true", help="Disable display windows")
    args = parser.parse_args()

    pipeline = CarlaStereoVSLAMPipeline(
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        fov=args.fov,
        baseline=args.baseline,
        show_display=not args.no_display
    )
    pipeline.set_weather(args.weather)

    # Spawn vehicle, attach sensors
    vehicle = pipeline.spawn_vehicle()
    pipeline.attach_stereo_cameras(vehicle)
    pipeline.attach_imu(vehicle)

    # Let the autopilot drive or do some scenario
    vehicle.set_autopilot(True)

    # Run SLAM
    pipeline.run_slam(duration=args.duration)

    # Cleanup
    pipeline.cleanup()


if __name__ == "__main__":
    main()
