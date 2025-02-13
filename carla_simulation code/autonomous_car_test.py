import carla
import random

def main():
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Load the world
    world = client.get_world()

    # Get a random spawn point
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)

    # Get the blueprint library
    blueprint_library = world.get_blueprint_library()

    # Select a random vehicle blueprint
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned vehicle: {vehicle.type_id} at {spawn_point.location}")

    # Attach a front-facing camera to the vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    # Position the camera on the vehicle
    camera_transform = carla.Transform(carla.Location(x=2.5, z=1.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Save images from the camera
    def process_image(image):
        image.save_to_disk('_out/%06d.png' % image.frame)
    camera.listen(process_image)

    # Move the spectator to follow the vehicle
    spectator = world.get_spectator()
    spectator_transform = carla.Transform(
        spawn_point.location + carla.Location(z=50),  # Move 50 units above the vehicle
        carla.Rotation(pitch=-90)  # Look straight down
    )
    spectator.set_transform(spectator_transform)
    print("Spectator moved to view the vehicle.")

    # Draw a debug box around the vehicle
    world.debug.draw_box(
        carla.BoundingBox(spawn_point.location, carla.Vector3D(2, 1, 1)),
        spawn_point.rotation,
        0.05,  # Line thickness
        carla.Color(255, 0, 0),  # Red box
        life_time=10.0  # Lasts for 10 seconds
    )
    print("Debug box drawn around the vehicle.")

    # Enable autopilot mode
    vehicle.set_autopilot(True)

    try:
        # Run the simulation for 15 seconds
        import time
        for _ in range(150):  # Adjust for a longer simulation
            spectator.set_transform(carla.Transform(
                vehicle.get_transform().location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            ))
            time.sleep(0.1)
    finally:
        # Clean up actors
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        print("Camera and vehicle destroyed.")

if __name__ == "__main__":
    main()
