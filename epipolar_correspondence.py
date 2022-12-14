import numpy as np
import cv2
import os


class EpipolarCorrespondence:
    def __init__(self,
                 step=10,
                 mean=0,
                 sigma=7,
                 F=None) -> None:
        self.step = step
        self.mean = mean
        self.sigma = sigma
        self.window_size = np.asarray([2*step+1, 2*step+1, 3]).astype(np.int16)
        self.gauss = self.create_gaussian()

    def create_gaussian(self):
        h, w, _ = self.window_size
        x, y = np.meshgrid(np.linspace(-1, 1, h),
                           np.linspace(-1, 1, w))
        dist = np.sqrt(x*x + y*y)
        gauss = np.exp(-(dist-self.mean)**2 / (2*self.sigma**2))    
        return gauss

    def window_score(self, window):
        '''
        window size: 2*step+1, 2*step+1, 3
        self.gauss: 2*step+1, 2*step+1 
        '''
        res = self.gauss.reshape(-1, 
                                 self.gauss.shape[1], 
                                 1)*window #h*w*3
        res = np.mean(res, axis=-1)
        return res

    def find_correspondence(self, 
                            img1,
                            img2,
                            pts1,
                            F):
        x1, y1 = pts1
        window1 = img1[y1-self.step: y1+self.step+1, 
                       x1-self.step: x1+self.step+1, 
                       :]
        gauss_window1 = self.window_score(window1)
        line_2 = F @ np.asarray([x1, y1, 1]).reshape(3, 1)
        a, b, c = line_2.reshape(-1)

        best_error = np.inf
        x2_req, y2_req = None, None

        for y2 in range(self.step, img2.shape[0]-self.step):
            x2 = int((-c-b*y2)/a)
            window2 = img2[y2-self.step: y2+self.step+1,
                           x2-self.step: x2+self.step+1, 
                           :]
            if window2.shape[1] != 2*self.step + 1:
                continue
            gauss_window2 = self.window_score(window2)
            curr_err = np.sum((gauss_window2 - gauss_window1)**2)
            if curr_err < best_error:
                best_error = curr_err
                x2_req = x2
                y2_req = y2

        return x2_req, y2_req