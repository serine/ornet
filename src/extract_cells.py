'''
extract_cells
This script will take in both a video from a directory and an
initial masking image of the first frame. It will then create contours for
each subsequent frame and use those contours to mask out each cell. It will
then save each cell in their own video
'''
# Author : Andrew Durden
import argparse
import os
import random
from functools import partial

import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt


def extract_cells(vidfile, maskfile, show_video=False):
    """
    reads video file and initial masks and returns a set of frames for each
    cell

    Parameters
    ----------
    vidfile : string
        path to a single video file
    maskfile : string
        path to a single vtk mask file
    show_video : boolean (Default : False)
        If true, display video with contours drawn during processing

    Returns
    ---------
    videos : Returns a list of arrays each with shape (H, W, F)
    """
    im = imageio.imread(maskfile)
    if(show_video):
        plt.imshow(im)
        plt.show()

    vf = imageio.get_reader(vidfile)

    # declare variables
    frameNum = 0
    number_of_segments = len(np.unique(im))-1  # define number of segs from vtk
    print(number_of_segments)
    outs = []
    ims = list()
    masks = list()
    colors = list()
    contours = list()
    dilates = list()
    comparisons = list()

    kernel = np.ones((17, 17), np.uint8)  # kernel for opening
    kernel2 = np.ones((3, 3), np.uint8)  # kernel for dilation
    font = cv2.FONT_HERSHEY_SIMPLEX  # font for frame count when show_video

    # determine random colors to use for the outlines in show_video
    for cols in range(number_of_segments):
        colors.append((random.randint(1, 255), random.randint(0, 255),
                       random.randint(1, 255)))

    # separates each mask from the vtk and lists them
    for i in range(number_of_segments):
            masks.append(im != i + 1)

    # for each frame
    for frameNum, frame in enumerate(vf):
        # create a copy of the current frame for each cell
        for i in range(number_of_segments):
            ims.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # mask out all that isn't in the initial/previous mask for each segment
        for i in range(number_of_segments):
            if(frameNum == 0):
                ims[i][masks[i]] = 0
            # use previous contour for next frame
            else:
                ims[i][dilates[i] != 255] = 0

        # delete previous masks once used for memory
        del dilates[:]

        # for each use threhold based segmentation to make strict contour
        for i in range(number_of_segments):
            ret, ims[i] = cv2.threshold(ims[i], 3, 255, cv2.THRESH_BINARY_INV)
            temp = cv2.morphologyEx(ims[i], cv2.MORPH_OPEN, kernel,
                                    iterations=3)
            temp = cv2.bitwise_not(temp)
            dilates.append(temp)

        """
        Iterative dilation for each cell with added overlap detection. For
        runtime a full pairwise overlap detection is done for each cell for the
        first 20 frames. During these frames, a lookup list us made of cells
        which overlap. After the 20 frames only the pairs in the lookup are
        checked, except every 100 frames where a full pairwise check is done
        again and the lookup is updated.
        """
        # full pairwise overlap detection
        if(frameNum % 100 == 1 or frameNum < 20):
            for i in range(5):
                del comparisons[:]
                for d in range(number_of_segments):
                    dilates[d] = cv2.dilate(dilates[d], kernel2, iterations=2)
                for s in range(number_of_segments):
                    newcomps = list()
                    for t in range(s+1, number_of_segments):
                        b_and = cv2.bitwise_and(dilates[s], dilates[t])
                        if(len(np.unique(b_and)) > 1):
                            newcomps.append(t)
                            dilates[s] = cv2.bitwise_xor(dilates[s], b_and)
                            dilates[t] = cv2.bitwise_xor(dilates[t], b_and)
                    comparisons.append(newcomps)
        # lookup list based overlap detection
        else:
            for i in range(5):
                for d in range(number_of_segments):
                    dilates[d] = cv2.dilate(dilates[d], kernel2, iterations=2)
                for s in range(number_of_segments):
                    for t in comparisons[s]:
                        b_and = cv2.bitwise_and(dilates[s], dilates[t])
                        if(len(np.unique(b_and)) > 1):
                            dilates[s] = cv2.bitwise_xor(dilates[s], b_and)
                            dilates[t] = cv2.bitwise_xor(dilates[t], b_and)

        # draws contours onto video for show_video
        for conts in range(number_of_segments):
            contours.append(cv2.findContours(dilates[conts], cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)[1])
            cv2.drawContours(frame, contours[conts], -1, colors[conts], 2)
            if (conts == number_of_segments - 1 and show_video):
                cv2.putText(frame, 'Frame # ' + str(frameNum), (10, 40), font,
                            0.5, (0, 255, 50), 1)
                cv2.imshow("Keypoints2", frame)
                k = cv2.waitKey(10)
                if (k == 27):
                    break

        # combines final contours into single frame of masks
        frameMask = np.zeros_like(dilates[0])
        for i in range(number_of_segments):
            frameMask[dilates[i] == 255] = i+1
        outs.append(frameMask)

        # free up resources
        del ims[:]
        del contours[:]
        del masks[:]

    # advanced video frames in show_video
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # returns list of mask frames
    return outs


if __name__ == '__main__':
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description="Read Video File and VTK File and creates videos of each"
                    "cell in the video",
        add_help="How to use", prog='extract_cells.py <args>')

    # Required args
    parser.add_argument("-i", "--input", required=True,
                        help="The path to a single video file")

    parser.add_argument("-m", "--masks", required=True,
                        help="The path to a single vtk file containing"
                             " first frame masks")

    # Optional args
    parser.add_argument("-o", "--output",
                        default=os.path.join(cwd, "single_cell_videos"),
                        help="Destination path for output videos."
                             " [Default: cwd/single_cell_videos]")

    parser.add_argument("-s", "--showvid", action="store_true",
                        help="If set, each frame with contours is drawn during"
                             " processing. [Default: False]")

    args = vars(parser.parse_args())

    # create output directory
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    vidpath = args['input']

    out = extract_cells(vidfile=args['input'], maskfile=args['masks'],
                        show_video=args['showvid'])
    fname = vidpath.split("/")[-1].split(".")[0]

    # write the files as npy files
    fname = "{}.npy".format(fname+'MASKS')
    outfile = os.path.join(args['output'], fname)
    np.save(outfile, out)
