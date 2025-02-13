import carla

# Connect to CARLA simulator
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world
world = client.get_world()

print("Connected to CARLA. Map:", world.get_map().name)
