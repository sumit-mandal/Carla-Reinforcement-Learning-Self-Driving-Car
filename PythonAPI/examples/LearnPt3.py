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
import math
import keras
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread
from tqdm import tqdm
from keras.callbacks import TensorBoard


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

SHOW_PREVIEW  = False #Show actual camera for carla. It shows what is happening with the environment when set to True
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000 # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 16 # How many steps (samples) to use for training
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5 #end of how many episode we want ot update target model
MODEL_NAME = 'Xception'

MEMORY_FRACTION = 0.8
MIN_REWARD = -200


EPISODES = 100

#exploration settings
DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

# Stats Setting
AGGREGATE_STATS_EVERY = 10


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

# Creating Environment Class
class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0 # Total 3 action -1,0,1
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None


    def __init__(self):
        self.client = carla.Client("localhost",2000)
        self.client.set_timeout(2.0)
        # Once we have client we can retrieve the world that is currently running
        self.world = self.client.get_world()

        #THe world contains the list blueprints that we can use for adding new actors into the simulation
        self.blueprint_library = self.world.get_blueprint_library()

        #Now let's filter all the blueprints of type 'vehicle' and choose on at random
        # print(blueprint_library.filter('vehicle'))
        self.model_3 = self.blueprint_library.filter('model3')[0] #spawning tesla model 3

    # Now we are going to create our reset method.
    def reset(self):
        self.collision_hist = [] # if store the data of collision.
        #if anything gets detected we are gonna say hey! You failed!!
        self.actor_list = [] # we are not gonna track actor always and will clean them at the end.

        self.transform = random.choice(self.world.get_map().get_spawn_points()) #where to spawn the car
        self.vehicle = self.world.spawn_actor(self.model_3,self.transform) #spawning the car

        self.actor_list.append(self.vehicle)

        # Get the blueprint of camera sensor
        self.rgb_cam = self.blueprint_library.find('sensor_camera.rgb')
        self.rgb_cam.set_attribute("image_size_x",f'{im_width}')
        self.rgb_cam.set_attribute("image_size_y",f'{im_height}')
        self.rgb_cam.set_attribute("fov",f"110")

        transform = carla.Transform(carla.Location(x=2.5,z=0.7)) #camera angle relative to car
        self.sensor = self.world.spawn_actor(self.rgb_cam,attach_to = self.vehicle) # this camera is than attached to vehicle
        self.actor_list.append(self.sensor)

        self.sensor.listen(lambda data:self.process_img(data)) #process_img is a fnction

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))
        time.sleep(4) # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        #Adding collision Method
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor,transform,attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event:collision_data(event)) #This is method defined below

        #Even after 4 seconds if our camera is  not ready then
        while self.front_camera is None:
            time.sleep(0.01)

        """we need to be certain the car is done falling from the sky on spawn.
         Finally, let's log the actual starting time for the episode, make sure
         brake and throttle aren't being used and return our first observation:"""

        self.episode_start = time.time() #The time function returns number of seconds passed since epoch
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))
        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    # we're going to take the data from the sensor, and pass it through some function called process_img.
    def process_img(image): #this process_img is in lambda
        i = np.array(image.raw_data)  # convert to an array #raw data is flattened array
        i2 = i.reshape((self.im_height,self.im_width,4)) #was flattened, so we're going to shape it.  #rbga that is why 4
        i3 = i2[:,:,:3] # # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
        if self.SHOW_CAM:
            cv2.imshow("show i3",i3)
            cv2.waitKey(10)
        self.front_camera = i3
        return i3/255.0 # normalising

    '''Now we need to do our step method. This method takes an action,
    and then returns the observation, reward, done, any_extra_info as per the
    usual reinforcement learning paradigm'''

    def step(self,action):
        if action == 0: # 0 is lest
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer = -1*self.STEER_AMT))
        elif action == 1: # 1 is left
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0,steer = 0*self.STEER_AMT))
        elif action == 2: #2 is right
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0,steer = 1*self.STEER_AMT))

        '''Above shows how we take an action based on what was passed as a
        numerical action to us, now we just need to handle for the observation,
        possible collision, and reward:'''

        v = self.vehicle.get_velocity #v = velocity
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) !=0: #if list of collision_hist is not blank. It means car have collided. THen give reward of -200
            done = True
            reward = -200

        elif kmh < 50: # else if car is running slowly and speed is less than 50 kmh. reward is -1
            done = False
            reward = -1

        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera,reward,done,None


        '''We know that every step an agent takes comes not only with a plausible prediction (depending on exploration/epsilon),
        but also with a fitment! This means that we're training and predicting at the same time, but it's pretty essential that
        our agent gets the most FPS (frames per second) possible. We'd also like our trainer to go as quick as possible.
        To achieve this, we can use either multiprocessing, or threading. Threading allows us to keep things still fairly simple.
        Later, we will likely at least open-source some multiprocessing code for this task at some point, but, for now, threading it is.'''

class DQNAgent():

    def __init__(self):
        self.model = self.create_model() # create_model is a def function
        self.target_model = self.create_model() ## Target model this is what we .predict against every step
        self.target_model.set_weights(self.model.get_weights()) #the weight that we get from self.model,set it to target_model


        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        #custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir =f"logs/{MODEL_NAME}-{int(time.time())}") # we don't want to update tensorboard per train session.

        self.target_update_counter = 0 #we use "self.target_update_counter" to decide when it's time to update our target model
        #(recall we decided update this model  every 'n' iterations, so that our predictions are reliable/stable).

        self.graph = tf.get_default_graph()

        self.terminate = False ## Should we quit?
        self.last_logged_episode = 0 # use to track tensorboard
        self.training_initialized = False #to track when TensorFlow is ready to get going.

    # Creating the model
    def create_model(self):
        # create the base pre-trained model
        base_model = Xception(weights=None, include_top = False, input_shape = (IM_HEIGHT,IM_WIDTH,3))

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # add a fully-connected layer
        predictions = Dense(3,activation='linear')(x) # 3 means left right and straight
        model = Model(inputs= base_model.input,outputs = predictions)
        model.compile(loss='mse',optimizer = Adam(lr=0.001),metrics=['accuracy'])
        return model

        #Here, we're just going to make use of the premade Xception model, but you could make some other model, or
        #import a different one. Note that we're adding GlobalAveragePooling to our ouput layer, as well as obviously
        # adding the 3 neuron output that is each possible action for the agent to take.


    #We need a quick method in our DQNAgent for updating replay memory:
    def update_replay_memory(self,transition):
        #transition is going to contain all the information we are gonna need to train the model
        # transition = (current_state,action,reward,new_state,done)
        self.replay_memory.append(transition)

    #Train Method. we only want to train if we have a bare minimum of samples in replay memory:
    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory,MINIBATCH_SIZE)  #If we don't have enough samples, we'll just return and be done.
        #If we do, then we will begin our training. First, we need to grab a random minibatch:

        #Once we have mini-batch we want to grab our current and furure q values
        current_states = np.array([transition[0] for transition in minibatch])/255 #transition[0] is current state
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states,PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/255 #transition[3] is future_state
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states,PREDICTION_BATCH_SIZE)

        # Now, we create our inputs (X) and outputs (y):
        X = []
        y = []

        for index,(current_state,action,reward,new_state,done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index]) #takes the maximum value from future_qs_list
                new_q = reward + DISCOUNT * max_future_q # if we are not done new_q = reward + discount * max_future_q
            else:
                new_q = reward # if we are done new_q is equal to reward

            current_qs = current_qs_list[index]
            print("current_qs",current_qs)
            current_qs[action] = new_q
            print("current_qs[action]",current_qs[action])

            X.append(current_state)
            y.append(current_qs)

        # In next steps ,We're only trying to log per episode, not actual training step, so we're going to use the following code to keep track.
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        #fitting the model

        with self.graph.as_default():
            self.model.fit(np.array(X)/255,np.array(y),batch_size = TRAINING_BATCH_SIZE ,verbose = 0,
            shuffle=False,callbacks=[self.tensorboard] if log_this_step else None ) # callbacks will be self.tensorboard if we are trying to log that step
            #otherwise no callbacks

        # we want ot continue tracking for logging
        if log_this_step:
            self.target_update_counter +=1

        if self.target_update_counter > UPDATE_TARGET_EVERY: # we want to update target model after 5 episodes
            self.target_model.set_weights(self.model.get_weights()) #after every 5 episode take the value from self.model and set into target_model
            self.target_update_counter = 0


    #we need a method to get q values (basically to make a prediction)

    def get_qs(self,state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0] # we are making prediction of states and unpacking it(height,width and 3)


    # Doing actual training
    def train_in_loop(self): #it uses concept of threading. We want to train and predict in different threads. We don't want it to slow down
        X = np.random.uniform(size = (1,IM_HEIGHT,IM_WIDTH,3)).astype(np.float32)
        y = np.random.uniform(size = (1,3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y,verbose = False,batch_size=1)

        self.training_initialized = True

        # To start, we use some random data like above to initialize, then we begin our infinite loop:
        while True:
            if self.terminate: #if it is true then trminate
                return
            self.train() #else train
            time.sleep(0.01)



if __name__=="main":
    FPS = 20
    #For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = MEMORY_FRACTION) #how much gpu memory our model will use
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))


    #Create models folder
    if not os.path.isdir("models"):#If this path doesn't exist then
        os.makedirs("models") #create folder models


     # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()


    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop(),daemon=True) #we want to train and predict in different thread
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    #Initialise predictions - First Prediction takes longer as of initialisation that has to be done
    #It's better to do a first prediction then before we start iterating oer episode steps.
    agent.get_qs(np.ones((env.im_height,env.im_width,3)))

    # start iterating over however many episodes we set to do:
    #Iterate over episode
    for episode in tqdm(range(1,EPISODES + 1),ascii=True,unit = 'episodes'):
        env.collision_hist = []
        #update tensorboard step every episode
        agent.tensorboard.step = episode #tensorboard by default sets evey frame as step.
        # We are resetting it to every episode to step

        #Reset episode - reset episode reward and step number
        episode_reward = 0
        step = 1
        # reset environment and initial state
        current_state = env.reset()
        #Reset flag and start iterating until episodes ends
        done = False
        episode_start = time.time()

        #Now  we are ready to run. Basically an environment will run until it's done, so we can use a While True loop and break on our done flag.

        #Play for given number of seconds
        while True: #Similar to if not done:
            if np.random.random() > epsilon:
                #Get action from q_table
                action =np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0,3)
                time.sleep(1/FPS)

        #Now we'll add our environment .step() method,which takes our action as a parameter.
            new_state,reward,done ,_ = env.step(action) #show new_state,reward,done ,_ based on action our agent takes
            episode_reward += reward #transform new continous state to new discrete state and count reward
            #Update replay_memory every step
            agent.update_replay_memory((current_state,action,reward,new_state,done)) #update with all the transition

            steps += 1

            if done:
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        # Now for some stats + saving models that have a good reward.
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            #Now let's decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON,epsilon)

        #if we've actually iterated through all of our target episodes, we can exit:
        # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')






































"""Reason for using two Networks
target_model is gonna be updated at the model. We want the model to be predict against to be relatively stable.
we kind of hold that model steady to predict against and then we are constantly training other model. And then after n
number of episode we want to update the model that we are predicting against(target_model).
Main model is update regularly and target models are updated after evey n step.

Every N steps, the weights from the main network are copied to the target network.

Here, you can see there are apparently two models: self.model and self.target_model.
What's going on here? So every step we take, we want to update Q values, but we also are
trying to predict from our model. Especially initially, our model is starting off as
random, and it's being updated every single step, per every single episode. What ensues
here are massive fluctuations that are super confusing to our model. This is why we almost
always train neural networks with batches (that and the time-savings). One way this is
solved is through a concept of memory replay, whereby we actually have two models.

The target_model is a model that we update every n episodes (where we decide on n), and this the model that
we use to determine what the future Q values.

Eventually, we converge the two models so they are the same,
but we want the model that we query for future Q values to be more stable than the model
that we're actively fitting every single step.

One network predicts the appropriate action and the second network predicts the target Q values for finding the Bellman error."""
