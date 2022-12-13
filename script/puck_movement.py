import torch
import numpy as np

b_params = np.load('../dyna_params/coll_params_b.npy')
n_params = np.load('../dyna_params/coll_params_n.npy')
theta_params = np.load('../dyna_params/coll_params_theta.npy')
b_params = torch.from_numpy(b_params).type(torch.FloatTensor)
n_params = torch.from_numpy(n_params).type(torch.FloatTensor)
theta_params = torch.from_numpy(theta_params).type(torch.FloatTensor)


def cross2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


class PuckMovement:
    def __init__(self, device):
        self.dampingX = 0.2125
        self.dampingY = 0.2562
        self.tableLength = 1.948
        self.tableWidth = 1.038
        self.puckRadius = 0.03165
        self.dt = 0.001
        self.device = device

        offsetP1 = torch.tensor([-self.tableLength / 2 + self.puckRadius, -self.tableWidth / 2 + self.puckRadius])
        offsetP2 = torch.tensor([-self.tableLength / 2 + self.puckRadius, self.tableWidth / 2 - self.puckRadius])
        offsetP3 = torch.tensor([self.tableLength / 2 - self.puckRadius, -self.tableWidth / 2 + self.puckRadius])
        offsetP4 = torch.tensor([self.tableLength / 2 - self.puckRadius, self.tableWidth / 2 - self.m_puckRadius])
        self.m_boundary = torch.tensor([[offsetP1[0], offsetP1[1], offsetP3[0], offsetP3[1]],
                                        [offsetP3[0], offsetP3[1], offsetP4[0], offsetP4[1]],
                                        [offsetP4[0], offsetP4[1], offsetP2[0], offsetP2[1]],
                                        [offsetP2[0], offsetP2[1], offsetP1[0], offsetP1[1]]], device=device)

        self.m_jacCollision = torch.eye(6, device=device)
        # transform matrix from global to local
        #   First Rim
        T_tmp = torch.eye(6, device=device)
        self.m_rimGlobalTransforms = torch.zeros((4, 6, 6), device=device)
        self.m_rimGlobalTransformsInv = torch.zeros((4, 6, 6), device=device)
        self.m_rimGlobalTransforms[0] = T_tmp
        self.m_rimGlobalTransformsInv[0] = torch.linalg.inv(T_tmp)
        #   Second Rim
        T_tmp = torch.zeros((6, 6), device=device)
        T_tmp[0][1] = 1
        T_tmp[1][0] = -1
        T_tmp[2][3] = 1
        T_tmp[3][2] = -1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[1] = T_tmp
        self.m_rimGlobalTransformsInv[1] = torch.linalg.inv(T_tmp)
        #   Third Rim
        T_tmp = torch.zeros((6, 6), device=device)
        T_tmp[0][0] = -1
        T_tmp[1][1] = -1
        T_tmp[2][2] = -1
        T_tmp[3][3] = -1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[2] = T_tmp
        self.m_rimGlobalTransformsInv[2] = torch.linalg.inv(T_tmp)
        #   Forth Rim
        T_tmp = torch.zeros((6, 6), device=device)
        T_tmp[0][1] = -1
        T_tmp[1][0] = 1
        T_tmp[2][3] = -1
        T_tmp[3][2] = 1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[3] = T_tmp
        self.m_rimGlobalTransformsInv[3] = T_tmp.T

    @property
    def non_coll_jac(self):
        J_linear = torch.eye(6, device=self.device)
        J_linear[0][2] = self.dt
        J_linear[1][3] = self.dt
        J_linear[2][2] = 1 - self.dt * self.tableDampingX
        J_linear[3][3] = 1 - self.dt * self.tableDampingY
        J_linear[4][5] = self.dt
        J_linear[5][5] = 1
        return J_linear

    def non_coll_dynamic(self, non_coll_state):
        return non_coll_state @ self.non_coll_jac.T

    def collision_in_boundary(self, s, r, pos):
        if ((self.m_boundary[2][:2] - pos > 0).all() and (self.m_boundary[0][:2] - pos < 0).all() and (
                s >= 1e-4 and s <= 1 - 1e-4 and r >= 1e-4 and r <= 1 - 1e-4)):
            return True
        else:
            return False

    # after one step prediction, check which is out of boundary, then which will collide
    # return indexs which are out of boundary
    def check_out_of_boundary(self,next_state_list):
        lr_index = torch.where(abs(next_state_list[:, 0]) > self.tableLength / 2)
        ud_index = torch.where(abs(next_state_list[:, 1]) > self.tableWidth / 2)
        index = torch.cat((lr_index[0], ud_index[0]))
        return torch.msort(index)

    def coll_dyna(self,coll_state):



    # main function of puck movement
    # first use check_coll to divide state list into coll and non_coll part; then use coll
    def forward(self,state):
        state_next = torch.zeros(state.shape)
        state_next = self.non_coll_dynamic(state)
        coll_index = self.check_out_of_boundary(state_next)
        coll_state_next = self.coll_dyna(state[coll_index])

