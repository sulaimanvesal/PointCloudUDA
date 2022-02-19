import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import pandas as pd
import os
# import argparse


def read_tf(tfrecord_path):
    """
    read in the tensors
    :param tfrecord_path: the path to the tensor
    :return: the image and the label
    """
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_path)
    # Create a dictionary describing the features.
    image_feature_description = {
        'data_vol': tf.io.FixedLenFeature([], tf.string),
        'label_vol': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    for parser in parsed_image_dataset:
        data_vol = tf.decode_raw(parser['data_vol'], tf.float32)

        label_vol = tf.decode_raw(parser['label_vol'], tf.float32)

        image_raw1 = data_vol.numpy()
        image_raw2 = label_vol.numpy()
        image_raw1 = image_raw1.reshape((256, 256, 3))
        image_raw2 = np.expand_dims(image_raw2.reshape((256, 256, 3))[..., 0], -1)
        return image_raw1, image_raw2


def tf_to_numpy(tf_path='../../input/'):
    """
    convert tensor to numpy array and save it
    :param tf_path: the path to the csv file that save all the path to the tensors
    :return:
    """
    for data_name in ["ct_train", "ct_val", "mr_train", "mr_val"]:
        df_train = pd.read_csv(os.path.join(tf_path, '{}_list.csv'.format(data_name)))
        ids_train = df_train['img']
        folder_tosave = os.path.join(tf_path, 'PnpAda_release_data/{}/'.format(data_name))
        if not os.path.exists(folder_tosave):
            os.mkdir(folder_tosave)
            if not os.path.exists(os.path.join(folder_tosave, 'img')):
                os.mkdir(os.path.join(folder_tosave, 'img/'))
            if not os.path.exists(os.path.join(folder_tosave, 'mask')):
                os.mkdir(os.path.join(folder_tosave, 'mask/'))
        for i, id in enumerate(ids_train):
            if i % 100 == 0:
                print(id)
            if not os.path.exists(os.path.join(folder_tosave, 'img', id)):
                img_path = '../../input/PnpAda_release_data/train_n_val/{}_tfs/{}'.format(data_name, id)
                img, mask = read_tf(img_path)
                np.save(os.path.join(folder_tosave, 'img', id), img)
                np.save(os.path.join(folder_tosave, 'mask', id), mask)
        print('**************** {} finished ****************'.format(data_name))

if __name__ == '__main__':
    # tf_to_numpy()
    # print("################ all the processes finished ################")
    img, mask = read_tf('../../input/PnpAda_release_data/train_n_val/ct_train_tfs/ct_train_slice{}.tfrecords'.format(0))
    print(img.shape, mask.shape)
    print(np.mean(img), np.std(img))
    print(img.min(), img.max())
    img2 = (img - img.min()) * 255 / (img.max() - img.min())
    img2 = np.array(img2, dtype=int)
    print(img2.min(), img2.max())
    from matplotlib import pyplot as plt
    plt.imshow(img2[128-112:128+112,128-112:128+112], cmap='gray')
    plt.show()
    plt.imshow(mask[128-112:128+112,128-112:128+112,0], cmap='gray')
    plt.show()

    img, mask = read_tf('../../input/PnpAda_release_data/train_n_val/ct_val_tfs/ct_val_slice{}.tfrecords'.format(1))
    print(img.shape, mask.shape)
    print(np.mean(img), np.std(img))
    print(img.min(), img.max())
    img2 = (img - img.min()) * 255 / (img.max() - img.min())
    img2 = np.array(img2, dtype=int)
    print(img2.min(), img2.max())
    plt.imshow(img2[128 - 112:128 + 112, 128 - 112:128 + 112], cmap='gray')
    plt.show()
    plt.imshow(mask[128 - 112:128 + 112, 128 - 112:128 + 112, 0], cmap='gray')
    plt.show()
