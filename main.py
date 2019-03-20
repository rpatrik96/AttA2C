import gym
#from model import ICMNet
from PIL import Image
import torchvision.transforms as transforms


# constants
num_epoch = 20
num_step = 100

# objects
env = gym.make('MsPacman-v0')
#icm_net = ICMNet(env.action_space.n)

# functions
def PixelsToTensor(pix):
    im2tensor = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Grayscale(1),
                                    transforms.Resize((42,42)),
                                    transforms.ToTensor()])

    return im2tensor(pix)

    

# aaannd ACTION!!!
for _ in range(num_epoch):
    obs = env.reset()

    for t in range(num_step):
        env.render()
        print(obs)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break