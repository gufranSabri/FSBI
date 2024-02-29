#program that goes through the cam figures and deletes the cam and original image if i click d
# the program reads the image using cv2, uses waitKey and does the necessary

import os
import cv2
import numpy as np

def main():
    path = "./fig/cam"
    images = os.listdir(path)
    images = [i for i in images if "cam" in i]
    images.sort()

    print(len(images))

    for i in images:
        print(i)
        img = cv2.imread(f"{path}/{i}")
        
        # img_dark = cv2.convertScaleAbs(img, alpha=2, beta=-100)
        # img = np.hstack((img, img_dark))

        cv2.imshow("Image", img)

        k = cv2.waitKey(0)
        if k == ord('d'):
            os.remove(f"{path}/{i}")
            os.remove(f"{path}/{i.replace('cam', 'original')}")
            print(f"Deleted {i} and {i.replace('cam', 'original')}")
        elif k == ord('q'):
            break


    # index = 0
    # while index < len(images):
    #     img = cv2.imread(f"{path}/{images[index]}")
    #     cv2.imshow("Image", img)
    #     k = cv2.waitKey(0)
    #     if k == ord('d'):
    #         os.remove(f"{path}/{images[index]}")
    #         os.remove(f"{path}/{images[index+1]}")
    #         index += 2
    #     elif k == ord('q'):
    #         break
    #     else:
    #         index += 2
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()