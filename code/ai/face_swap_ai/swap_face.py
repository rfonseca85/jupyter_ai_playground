import cv2
import matplotlib.pyplot as plt

def swap_n_show(img1, img1_pos, img2, img2_pos, app, swapper, plot_before=True, plot_after=True):
    # read the 2 images
    img1 = cv2.imread('img/' + img1)
    img2 = cv2.imread('img/' + img2)

    if plot_before:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1[:, :, ::-1])
        axs[0].axis('off')
        axs[1].imshow(img2[:, :, ::-1])
        axs[1].axis('off')
        plt.show()

        # Do the swap
        face1 = app.get(img1)[img1_pos]
        bbox = face1['bbox']
        bbox = [int(b) for b in bbox]
        plt.imshow(img1[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
        plt.show()

        face2 = app.get(img2)[img2_pos]
        bbox = face2['bbox']
        bbox = [int(b) for b in bbox]
        plt.imshow(img2[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
        plt.show()

    img1 = img1.copy()
    img2 = img2.copy()

    if plot_after:
        img2 = swapper.get(img2, face1, face2, paste_back=True)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img2[:, :, ::-1])
        axs[0].axis('off')

        face_final = app.get(img2)[img2_pos]
        bbox = face_final['bbox']
        bbox = [int(b) for b in bbox]
        plt.imshow(img2[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
        plt.show()

    return img1, img2