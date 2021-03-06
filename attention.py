import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class MaskContainer():

    def __init__(self, mask_file=None):
        if mask_file is not None:
            self._masks = np.load(mask_file)

    def get_mask_iter(self):
        return iter(self._masks)

    def get_mask(self, i):
        return self._masks[i]

    def save_masks(self, fname):
        np.save(fname, self._masks)


class Attention(MaskContainer):

    def __init__(self, tracker, scale=1.0):
        super().__init__()
        self.tracker = tracker
        self.H, self.W = tracker.H, tracker.W
        self.scale = scale
        self.generate_masks()

    def generate_masks(self):
        self._masks = []
        xi = np.arange(self.H)
        xj = np.arange(self.W)

        for x, y, r in self.tracker.get_pts():
            sd = r * self.scale
            sdi = sd
            sdj = sd * (self.W / self.H)
            pi = stats.norm(y, sdi)
            pj = stats.norm(x, sdj)
            i = sdi * pi.pdf(xi).reshape(1, -1)
            j = sdj * pj.pdf(xj).reshape(1, -1)

            thresh = (i.T @ j) * 2 * np.pi
            rnd = np.random.uniform(0, 1, size=thresh.shape)
            mask = np.uint8(rnd < thresh)
            self._masks.append(mask)

        self._masks = np.stack(self._masks)


class Tracker():
    DEFAULT_SD = 20

    def __init__(self, episode_path):
        arr = np.load(episode_path)
        _, self.H, self.W, _ = arr.shape

        self._pts = []
        for frame in arr:
            pts = self.find_points(frame)
            self._pts.append(pts[0])

    def get_pts(self):
        return self._pts

    def find_points(self, frame):
        raise NotImplementedError("Default Tracker class has no find_points method")


class CubeTracker(Tracker):

    def __init__(self, episode_path, debug=False):
        self.debug = debug
        super().__init__(episode_path)

    def show_rgb(self, im):
        bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow('im', bgr)
        cv2.waitKey(0)

    def find_points(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.medianBlur(gray,11)
        ret, thresh = cv2.threshold(blur, 100, 255, 1)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        if len(contours) == 0:
            return [(self.H//2, self.W//2, self.DEFAULT_SD)]

        (x, y), r = cv2.minEnclosingCircle(contours[0])
        cx, cy = (int(x), int(y))

        if self.debug:
            cv2.drawContours(frame, contours, -1, (0,255,0), 3)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            cv2.circle(frame, (cx, cy), int(r), (255, 0, 0), 2)
            self.show_rgb(frame)

        return [(cx, cy, r/2)]


if __name__ == '__main__':
    tracker = CubeTracker('saved_episodes/1_noisy.npy', debug=True)
    attn = Attention(tracker)
    for i, mask in enumerate(attn.get_mask_iter()):
        plt.imsave(f'masks/{i:03d}.png', mask)
        print(mask.sum()/mask.size)

