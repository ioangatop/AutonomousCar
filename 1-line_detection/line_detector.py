import cv2
import numpy as np

from sklearn.linear_model import HuberRegressor, Ridge


class LineDetector:
    def __init__(self, **kargs):
        self.gaussian_kernel = kargs['gaussian_kernel']
        self.low_threshold, self.high_threshold = kargs['canny_thresholds']
        self.hough_settings = kargs['hough_settings']
        self.line_threshold = kargs['line_threshold']
        self.line_thickness = kargs['line_thickness']

    def grayscale(self, img):
        """Applies the Grayscale transform"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        """Applies an image mask."""
        mask = np.zeros_like(img)   
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(img, mask)


    # def draw_lines(self, img, lines):
    #     """Averages/extrapolates the detected line segments"""
    #     line_dict = {'left':[], 'right':[]}
    #     img_center = img.shape[1]//2
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             a = (y2-y1)/(x2-x1)
    #             x_mean, y_mean = (x1+x2)/2, (y1+y2)/2
    #             position = 'left' if a <= 0 else 'right'
    #             line_dict[position].append(np.array([a, x_mean, y_mean]))
    #             line_dict[position].append(np.array([a, x_mean, y_mean]))

    #     for position, lines_dir in line_dict.items():
    #         data = np.array(lines_dir)
    #         data = data[data[:, 2] >= np.array(self.line_threshold)-1]
    #         coef_ = np.median(data[:,0])
    #         x = np.median(data[:,1])
    #         y = np.median(data[:,2])
    #         intercept_ = y - x*coef_

    #         epsilon = 1e-10
    #         y1 = np.array(img.shape[0])
    #         x1 = int((y1 - intercept_)/(coef_+epsilon))
    #         y2 = np.array(self.line_threshold)
    #         x2 = int((y2 - intercept_)/(coef_+epsilon))

    #         cv2.line(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=self.line_thickness)

    def draw_lines(self, img, lines):
        """Averages/extrapolates the detected line segments"""
        line_dict = {'left':[], 'right':[]}
        img_center = img.shape[1]//2
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1<img_center and x2<img_center:
                    position = 'left'
                elif x1>img_center and x2>img_center:
                    position = 'right'
                else:
                    continue
                line_dict[position].append(np.array([x1, y1]))
                line_dict[position].append(np.array([x2, y2]))

        for position, lines_dir in line_dict.items():
            data = np.array(lines_dir)
            data = data[data[:, 1] >= np.array(self.line_threshold)-1]
            x, y = data[:, 0].reshape((-1, 1)), data[:, 1]

            try:
                model = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,
                                    epsilon=1.9)
                model.fit(x, y)
            except ValueError:
                model = Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True)
                model.fit(x, y)


            epsilon = 1e-10
            y1 = np.array(img.shape[0])
            x1 = (y1 - model.intercept_)/(model.coef_+epsilon)
            y2 = np.array(self.line_threshold)
            x2 = (y2 - model.intercept_)/(model.coef_+epsilon)
            x = np.append(x, [x1, x2], axis=0)

            # cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=self.line_thickness)

            y_pred = model.predict(x)
            data = np.append(x, y_pred.reshape((-1, 1)), axis=1)
            cv2.polylines(img, np.int32([data]), isClosed=False,
                          color=(255, 0, 0), thickness=self.line_thickness)

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """Returns an image with hough lines drawn."""
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self.draw_lines(line_img, lines)
        return line_img

    def weighted_img(self, img, initial_img, α=0.95, β=1., γ=0.):
        """initial_img * α + img * β + γ"""
        return cv2.addWeighted(initial_img, α, img, β, γ)

    def forward(self, img):
        """Detect lines and draw them on the image"""
        img_line = self.grayscale(img)
        img_line = self.gaussian_blur(img_line, kernel_size=self.gaussian_kernel)
        img_line = self.canny(img_line, low_threshold=self.low_threshold,
                              high_threshold=self.high_threshold)

        vertices = np.array([[(0, img_line.shape[0]), (img_line.shape[1], img_line.shape[0]),
                              (400, 260), (600, 260)]])

        img_line = self.region_of_interest(img_line, vertices)
        img_line = self.hough_lines(img_line, **self.hough_settings)

        return self.weighted_img(img_line, img)

if __name__ == "__main__":
    pass
