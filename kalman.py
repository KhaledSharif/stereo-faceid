import numpy as np

class KalmanFilter3D(object):
    def __init__(self, mean_x, mean_y, mean_z):
        self.F = np.array([

            # x dimension
            [1., 1., 0., 0., 0., 0., 0., 0., 0.],  # 1x position + 1x velocity
            [0., 1., 1., 0., 0., 0., 0., 0., 0.],  # 1x velocity + 1x acceleration
            [0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 1x acceleration

            # y dimension
            [0., 0., 0., 1., 1., 0., 0., 0., 0.],  # 1x position + 1x velocity
            [0., 0., 0., 0., 1., 1., 0., 0., 0.],  # 1x velocity + 1x acceleration
            [0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 1x acceleration

            # z dimension
            [0., 0., 0., 0., 0., 0., 1., 1., 0.],  # 1x position + 1x velocity
            [0., 0., 0., 0., 0., 0., 0., 1., 1.],  # 1x velocity + 1x acceleration
            [0., 0., 0., 0., 0., 0., 0., 0., 1.],  # 1x acceleration

        ]).astype(float)  # next state function

        N = self.F.shape[1]

        self.H = np.array([
            [1., 0., 0., 0., 0., 0., 0., 0., 0.], # measure x position
            [0., 0., 0., 1., 0., 0., 0., 0., 0.], # measure y position
            [0., 0., 0., 0., 0., 0., 1., 0., 0.], # measure z position
        ]).astype(float)  # measurement function

        assert self.H.shape == (3, 9)

        self.R = np.eye(3).astype(float)
        self.I = np.eye(N).astype(float)  # identity matrix

        # initial uncertainty
        self.P = np.eye(N).astype(float) * 1000.0

        # initial estimate of position, velocity, acceleration
        self.x = np.array([[mean_x, 0, 0, mean_y, 0, 0, mean_z, 0, 0]]).reshape(9, 1).astype(float)

        assert self.F.shape[1] == self.x.shape[0]
        assert self.H.shape[1] == self.x.shape[0]

    def update(self, measurement_x, measurement_y, measurement_z):
        assert self.H.shape[1] == self.x.shape[0]
        
        Z = self.H @ self.x
        y = np.array([measurement_x, measurement_y, measurement_z]).reshape(3, 1) - Z
        S = self.H @ self.P @ self.H.T
        S += self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T
        x, y, z = self.x[0][0], self.x[3][0], self.x[6][0]  # x, y, z coordinates

        assert isinstance(x, float), type(x)
        assert isinstance(y, float), type(y)
        assert isinstance(z, float), type(z)
        
        return x, y, z


