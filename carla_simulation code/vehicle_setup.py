import carla
import time
import numpy as np
import cv2
import open3d as o3d  # For LiDAR visualization
import threading

class VehicleSetup:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.sensors = {}
        # self.list_available_vehicles()

    def list_available_vehicles(self):
        
        available_vehicles = [bp.id for bp in self.blueprint_library.filter('vehicle.*')]
        print("\n[INFO] Available Vehicles in CARLA:")
        for vehicle in available_vehicles:
            print(vehicle)
        print("\n")

    
    def spawn_vehicle(self):
        """Spawns a Dodge Police Charger 2020 in CARLA"""
        available_vehicles = [bp.id for bp in self.blueprint_library.filter('vehicle.*')]
        vehicle_bp = self.blueprint_library.find(available_vehicles[0]) 
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"[INFO] Vehicle {vehicle_bp.id} spawned successfully!")

        # Set spectator (Bird's Eye View)
        self.set_spectator_camera()

    def attach_sensors(self):
        """Attaches required sensors to the vehicle"""
        if not self.vehicle:
            print("[ERROR] No vehicle found. Call spawn_vehicle() first.")
            return

        # ========== Front RGB Camera ==========
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.sensors['front_camera'] = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.sensors['front_camera'].listen(lambda image: self.show_image(image, "Front Camera"))

        # ========== LiDAR Sensor ==========
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')  # 50 meters range
        lidar_bp.set_attribute('points_per_second', '500000')  # High-resolution
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))  # Roof-mounted
        self.sensors['lidar'] = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.sensors['lidar'].listen(self.show_lidar)

        print("[INFO] Sensors attached successfully!")

    def set_spectator_camera(self):
        """Moves the spectator camera above the vehicle for a bird’s eye view"""
        spectator = self.world.get_spectator()
        if self.vehicle is None:
            print("[ERROR] Vehicle not found! Cannot set spectator.")
            return

        def update_spectator():
            """Updates spectator camera position to follow vehicle"""
            while True:
                vehicle_transform = self.vehicle.get_transform()
                vehicle_location = vehicle_transform.location
                new_location = carla.Location(vehicle_location.x, vehicle_location.y, vehicle_location.z + 15)  # Set height
                new_rotation = carla.Rotation(pitch=-90, yaw=vehicle_transform.rotation.yaw, roll=0)  # Look directly down
                spectator.set_transform(carla.Transform(new_location, new_rotation))
                time.sleep(0.1)  # Update every 100ms

        thread = threading.Thread(target=update_spectator, daemon=True)
        thread.start()
        print("[INFO] Spectator camera set to bird's eye view.")

    def show_image(self, image, window_name):
        """Displays images from the sensors in OpenCV"""
        image.convert(carla.ColorConverter.Raw)
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img_array.reshape((image.height, image.width, 4))[:, :, :3]
        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    def show_lidar(self, point_cloud):
        """Displays LiDAR point cloud"""
        points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]

        # Convert to Open3D Point Cloud format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Visualize in Open3D
        o3d.visualization.draw_geometries([pcd], window_name="LiDAR Point Cloud")

    def get_sensor(self, sensor_name):
        """Returns the requested sensor instance"""
        return self.sensors.get(sensor_name, None)

    def destroy(self):
        """Destroys all actors when exiting"""
        for sensor in self.sensors.values():
            sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        print("[INFO] Vehicle and sensors destroyed.")

# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    setup = VehicleSetup()
    setup.spawn_vehicle()
    setup.attach_sensors()

    try:
        while True:
            pass  # Keep simulation running
    except KeyboardInterrupt:
        pass
    finally:
        setup.destroy()
