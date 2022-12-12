import torch as torch
import numpy as np


class Controller:
    def __init__(self):
        self.kp = 50
        self.kv = 20
        self.pos = None
        self.vel = None

    def reset(self, mallet_pos, mallet_vel, kp=None, kv=None):
        self.pos = mallet_pos
        self.vel = mallet_vel
        if kp is not None:
            self.kp = kp
        if kv is not None:
            self.kv = kv

# calculate action u for each time step
# current just simple pd controller
    def pd_control(self, puck_pos, puck_vel):
        u = self.kp * (puck_pos - self.pos) + self.kv * (puck_vel - self.vel)
        return u