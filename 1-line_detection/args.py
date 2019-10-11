import argparse
import math

def parser():
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--path', default='test_videos/', type=str,
                        help='Folder path of the videos.')
    PARSER.add_argument('--file_', default='solidWhiteRight.mp4', type=str,
                        help='Full name of the file.')
    PARSER.add_argument('--subclip', default=20, type=int)

    PARSER.add_argument('--gaussian_kernel', default=5, type=int)
    PARSER.add_argument('--canny_thresholds', default=[200, 300], type=list)
    PARSER.add_argument('--line_threshold', default=330, type=float)
    PARSER.add_argument('--line_thickness', default=6, type=float)
    PARSER.add_argument('--rho', default=1, type=int)
    PARSER.add_argument('--theta', default=math.pi/180, type=float)
    PARSER.add_argument('--threshold', default=15, type=int)
    PARSER.add_argument('--min_line_len', default=30, type=float)
    PARSER.add_argument('--max_line_gap', default=40, type=float)

    ARGS = PARSER.parse_args()
    return ARGS

args = parser()

settings = {'gaussian_kernel':args.gaussian_kernel,
            'canny_thresholds':args.canny_thresholds,
            'line_threshold':args.line_threshold,
            'line_thickness':args.line_thickness,
            'hough_settings':{'rho':args.rho,
                              'theta':args.theta,
                              'threshold':args.threshold,
                              'min_line_len':args.min_line_len,
                              'max_line_gap':args.max_line_gap}}

if __name__ == "__main__":
    pass
