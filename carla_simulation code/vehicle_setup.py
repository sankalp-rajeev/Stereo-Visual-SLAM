import carla
import time
import numpy as np
import cv2
import open3d as o3d
import threading

SHOW_SENSOR_OUTPUT = True

class VehicleSetup:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.sensors = {}

        # A place to store the latest LiDAR points (thread-safe variable).
        self.lidar_points = None

    def spawn_vehicle(self):
        """Spawns a vehicle in CARLA and enables autopilot."""
        # For reliability, pick a known blueprint instead of [0]
        available_vehicles = [bp.id for bp in self.blueprint_library.filter('vehicle.*')]
        vehicle_bp = self.blueprint_library.find(available_vehicles[0])
        spawn_points = self.world.get_map().get_spawn_points()

        for i, spawn_point in enumerate(spawn_points):
            try:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle:
                    print(f"[INFO] Vehicle {vehicle_bp.id} spawned at point {i}")
                    self.vehicle.set_simulate_physics(True)
                    self.vehicle.set_autopilot(False)

                    time.sleep(1)

                    # Traffic Manager
                    self.tm = self.client.get_trafficmanager(8000)
                    settings = self.world.get_settings()
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                    self.world.apply_settings(settings)

                    self.tm.set_synchronous_mode(True)

                
                    # If ignoring signs/lights causes frequent crashes, set these to 0
                    self.tm.ignore_lights_percentage(self.vehicle, 0)
                    self.tm.ignore_signs_percentage(self.vehicle, 0)
                    self.tm.auto_lane_change(self.vehicle, True)
                    self.tm.set_desired_speed(self.vehicle, 20)  # reduce speed from 50 to 30
                    self.tm.distance_to_leading_vehicle(self.vehicle, 5.0)

                    time.sleep(1)
                    self.vehicle.set_autopilot(True, self.tm.get_port())
                    print("[INFO] Autopilot enabled.")

                    self.attach_sensors()
                    self.set_spectator_camera()
                    return
            except RuntimeError as e:
                print(f"[WARNING] Spawn attempt {i} failed: {e}")

        print("[ERROR] All spawn attempts failed! No free spawn points available.")
        exit(1)

    def attach_sensors(self):
        """Attaches camera, lidar, IMU, GNSS to the vehicle."""
        if not self.vehicle:
            print("[ERROR] No vehicle found. Call spawn_vehicle() first.")
            return

        # 1) Front RGB Camera
        self.add_camera('front_camera', carla.Location(x=1.5, z=2.0),
                        width='800', height='600', fov='90')

        # 2) LiDAR
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('points_per_second', '1000000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        lidar_actor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        lidar_actor.listen(self.lidar_callback)
        self.sensors['lidar'] = lidar_actor

        # 3) IMU Sensor
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0, z=1.0))
        imu_actor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        imu_actor.listen(lambda imu: print(f"[IMU] Accel: {imu.accelerometer}, Gyro: {imu.gyroscope}")
                         if SHOW_SENSOR_OUTPUT else None)
        self.sensors['imu'] = imu_actor

        # 4) GNSS Sensor
        gnss_bp = self.blueprint_library.find('sensor.other.gnss')
        gnss_transform = carla.Transform(carla.Location(x=0, z=2.0))
        gnss_actor = self.world.spawn_actor(gnss_bp, gnss_transform, attach_to=self.vehicle)
        gnss_actor.listen(lambda gnss: print(f"[GNSS] Lat: {gnss.latitude}, Lon: {gnss.longitude}")
                          if SHOW_SENSOR_OUTPUT else None)
        self.sensors['gnss'] = gnss_actor

        print("[INFO] Sensors attached successfully!")

    def add_camera(self, name, transform, width, height, fov, sensor_type='sensor.camera.rgb'):
        """Helper function to add cameras."""
        camera_bp = self.blueprint_library.find(sensor_type)
        camera_bp.set_attribute('image_size_x', width)
        camera_bp.set_attribute('image_size_y', height)
        camera_bp.set_attribute('fov', fov)
        camera_actor = self.world.spawn_actor(camera_bp, carla.Transform(transform), attach_to=self.vehicle)
        camera_actor.listen(lambda img: self.camera_callback(img, name))
        self.sensors[name] = camera_actor

    def camera_callback(self, image, name):
        """Handles camera feed in OpenCV windows."""
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img_array.reshape((image.height, image.width, 4))[:, :, :3]
        if SHOW_SENSOR_OUTPUT:
            cv2.imshow(name, img)
            cv2.waitKey(1)

    def lidar_callback(self, point_cloud):
        """Stores LiDAR points in a thread-safe variable; do not visualize here!"""
        points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        self.lidar_points = points

    def set_spectator_camera(self):
        """Puts the Unreal 'Spectator' camera overhead and updates it in a thread."""
        spectator = self.world.get_spectator()

        if self.vehicle:
            # Immediate set
            vehicle_transform = self.vehicle.get_transform()
            spectator_transform = carla.Transform(
                vehicle_transform.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            )
            spectator.set_transform(spectator_transform)
            print("[INFO] Spectator camera set at vehicle spawn location.")

        # Continuous update in background
        def follow_vehicle():
            while True:
                self.world.tick()
                if not self.vehicle:
                    break
                vehicle_transform = self.vehicle.get_transform()
                spectator_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(z=50),
                    carla.Rotation(pitch=-90)
                )
                spectator.set_transform(spectator_transform)
                time.sleep(0.05)

        spectator_thread = threading.Thread(target=follow_vehicle, daemon=True)
        spectator_thread.start()
        print("[INFO] Spectator camera is now following the vehicle.")

def main():
    setup = VehicleSetup()
    setup.spawn_vehicle()

    # Create the Open3D Visualizer in the main thread
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name='LiDAR Point Cloud', width=800, height=600)
    # pcd = o3d.geometry.PointCloud()
    # vis.add_geometry(pcd)

    try:
        while True:
            
            # If we have new lidar points, update them in the Visualizer
            
            # if setup.lidar_points is not None:
            #     pcd.points = o3d.utility.Vector3dVector(setup.lidar_points)
            #     vis.update_geometry(pcd)

            # # Process Open3D events (non-blocking)
            # vis.poll_events()
            # vis.update_renderer()

            # Let OpenCV camera windows update as well
            cv2.waitKey(1)

            # Avoid busy looping
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping Simulation...")
    finally:
        if setup.vehicle is not None:
            setup.vehicle.destroy()

        cv2.destroyAllWindows()
        # vis.destroy_window()

if __name__ == "__main__":
    main()
