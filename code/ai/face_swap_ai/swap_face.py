import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis

def swap_n_show(img1, img1_pos, img2, img2_pos, plot_before=True):

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('model/inswapper_128.onnx', download=False, download_zip=False)

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


    res = img2.copy()

    res = swapper.get(res, face2, face1, paste_back=True)
    plt.imshow(res[:, :, ::-1])
    plt.show()

    face_final = app.get(res)[img2_pos]
    bbox = face_final['bbox']
    bbox = [int(b) for b in bbox]
    plt.imshow(res[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
    plt.show()

    return res
