"""Actor: Actor is anything that plays a role in the simulation and can be moved around, examples of actors are vehicles, pedestrians, and sensors.
Blueprint: Before spawning an actor you need to specify its attributes, and that's what blueprints
are for. We provide a blueprint library with the definitions of all the actors available.
World: The world represents the currently loaded map and contains the functions for converting a
blueprint into a living actor, among other. It also provides access to the road map and functions to change the weather conditions."""

import glob
import os
import sys
import random
import time
import numpy as np
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

"""The first thing we'll take care of immediately is the list of actors, and cleaning them up
when we're done. Recall that we have both a client and server. When we start running a client on
 the server, we create actors on the server. If we just exit, without cleaning up, our actors will
 still be on the server."""

"""To connect to a simulator we need to create a "Client" object, to do so we need to provide the
 IP address and port of a running instance of the simulator"""

"""The first recommended thing to do right after creating a client instance is setting its time-out.
 This time-out sets a time limit to all networking operations, if the time-out is not set networking
 operations may block forever"""

IM_WIDTH = 640 #open cv window of the width 640
IM_HEIGHT = 480 #open cv window of the height 480

#we're going to take the data from the sensor, and pass it through some function called process_img.
def process_img(image): #this process_img is in lambda
    i = np.array(image.raw_data)  # convert to an array #raw data is flattened array
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4)) #was flattened, so we're going to shape it.  #rbga that is why 4
    i3 = i2[:,:,:3] # # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
    cv2.imshow("show i3",i3)
    cv2.waitKey(10)
    return i3/255.0 # normalising

actor_list = []
try:
    client = carla.Client('localhost',2000)
    client.set_timeout(2.0)

    world = client.get_world() #Once we have the client configured we can directly retrieve the world
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("model3")[0] #grabbing tesla model3 at 0th index
    print(bp)

    #where to spawn the car
    spawn_point = random.choice(world.get_map().get_spawn_points())
    #spawning the car
    vehicle = world.spawn_actor(bp,spawn_point)
    # vehicle.set_autopilot(True)

    vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0.0)) #Vehicles are a special type of
    # actor that provide a few extra methods. Apart from the handling methods common to all actors, vehicles can also
    # be controlled by providing throttle, break, and steer values
    actor_list.append(vehicle) #appen this vehicle intoa actor_list

    #Get the blueprinnt for this sensor
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    #CHange the dimensions of the image
    cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
    cam_bp.set_attribute('fov','110')

    """Next, we need to add this to our car. First, we'll adjust the sensor
    from a relative position, then we'll attach this to our car. So we'll say this sensor,
    from it's relative position (the car), we want to move forward 2.5 and up 0.7.
    I don't know if this is in meters or what"""

    #Adjust sensor realtive to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5,z = 0.7)) #x is forward y is left and right, z is up and down

    #Spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(cam_bp,spawn_point,attach_to = vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    #we want to do something with this sensor. We want to get the imagery from it, so we want to listen.
    sensor.listen(lambda data:process_img(data)) #this process_img is def_fuction


    time.sleep(10)

finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up")
