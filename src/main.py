import gym
from agent import ICMAgent
from PIL import Image
import torchvision.transforms as transforms



# constants
num_epoch = 5
num_step = 50

# objects
#env = gym.make('MsPacman-v0')
env = gym.make('MontezumaRevenge-v0')
agent = ICMAgent(env.action_space.n)

agent.cuda()
agent.train('MontezumaRevenge-v0', num_epoch, num_step)
