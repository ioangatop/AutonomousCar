import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from line_detector import LineDetector
from args import args, settings


def main():
    line_finder = LineDetector(**settings)
    if args.file_.split('.')[-1] == 'jpg':
        img = mpimg.imread(args.path+args.file_)
        output_img = line_finder.forward(img)
        plt.imshow(output_img)
        plt.show()
    elif args.file_.split('.')[-1] == 'mp4':
        white_output = 'test_videos_output/output_video.mp4'
        clip1 = VideoFileClip(args.path+args.file_).subclip(0, args.subclip)
        white_clip = clip1.fl_image(line_finder.forward)
        white_clip.write_videofile(white_output, audio=False)
    else:
        print('Error {}: Not supported type.'.format(args.file_.split('.')[-1]))
        quit()

if __name__ == "__main__":
    main()
