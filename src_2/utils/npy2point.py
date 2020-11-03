import os, glob
import numpy as np
from tqdm import tqdm
import argparse


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def graipher(pts, K, dim=2):
    farthest_pts = np.zeros((K, dim))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def crop_volume(vol, crop_size=112):

    """
    :param vol:
    :return:
    """

    return np.array(vol[:,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                    int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])


def npy2point_test(folder='ct_train', to_save='v', number_points=300, dim=3, crop_size=224, tocrop=False):
    assert to_save == '' or to_save == 'v' or to_save == 'p', "'v' represents vertices, 'p' represents plots, '' represents all."
    import mcubes
    crop_from = 128 - crop_size // 2
    crop_to = 128 + crop_size // 2
    vertices_fold = os.path.join('../../input/PnpAda_release_data/', folder, 'vertices/')
    plots_fold = os.path.join('../../input/PnpAda_release_data/', folder, 'plots/')
    if not os.path.exists(vertices_fold):
        os.mkdir(vertices_fold)
    if not os.path.exists(plots_fold):
        os.mkdir(plots_fold)
    folder_path = os.path.join('../../input/PnpAda_release_data/', folder, "mask/")
    path = os.path.join(folder_path, "mr_val_slice509.tfrecords.npy")
    filename = os.path.splitext(os.path.basename(path))[0]
    vertices_path = os.path.join(vertices_fold, filename + '.npy')
    plot_path = os.path.join(plots_fold, filename + '.npy')
    if not os.path.exists(vertices_path):
        mask = np.load(path)
        if args.toplot:
            from matplotlib import pyplot as plt
            temp = mask[..., 0][crop_from:crop_to, crop_from:crop_to] if tocrop else mask[..., 0]
            plt.imshow(temp)
            plt.show()
        mask = np.where(mask > 0, 1, 0)
        mask = np.moveaxis(mask, -1, 0)
        if tocrop:
            mask = crop_volume(mask, crop_size=crop_size)
        mask = np.concatenate([mask, mask, mask], axis=0)
        point_cloud = np.zeros((crop_size, crop_size)) if tocrop else np.zeros((256, 256))
        vertices_array = np.zeros((number_points, dim))
        if mask.sum() > 10:
            vol = mcubes.smooth(mask)
            vertices, triangles = mcubes.marching_cubes(vol, 0)
            try:
                vertices = graipher(vertices, number_points, dim=dim)
            except:
                print(filename)
                exit()
            vertices_array = np.array(vertices, dtype=np.int)
            if args.toplot:
                fig = plt.figure()
                from mpl_toolkits.mplot3d import Axes3D
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=10)
                plt.show()
            point_cloud[vertices_array[:, 1], vertices_array[:, 2]] = 1
            if args.toplot:
                plt.imshow(point_cloud, cmap='gray')
                plt.show()

        if to_save == 'v' or to_save == '':
            np.save(vertices_path, vertices_array)
        if to_save == 'p' or to_save == '':
            np.save(plot_path, point_cloud)
            # mcubes.export_mesh(vertices, triangles, "heart_single_slice.dae", "MyHeart_s")
    print("finish")


def npy2point(folder='ct_train', to_save='v', number_points=300, dim=3, crop_size=224, tocrop=False):
    assert to_save=='' or to_save=='v' or to_save=='p', "'v' represents vertices, 'p' represents plots, '' represents all."
    import mcubes
    crop_from = 128 - crop_size//2
    crop_to = 128 + crop_size//2
    vertices_fold = os.path.join('../../input/PnpAda_release_data/', folder, 'vertices/')
    plots_fold = os.path.join('../../input/PnpAda_release_data/', folder, 'plots/')
    if not os.path.exists(vertices_fold):
        os.mkdir(vertices_fold)
    if not os.path.exists(plots_fold):
        os.mkdir(plots_fold)
    folder_path = os.path.join('../../input/PnpAda_release_data/', folder, "mask/")
    for path in tqdm(glob.glob(folder_path + '*.npy')):
        filename = os.path.splitext(os.path.basename(path))[0]
        vertices_path = os.path.join(vertices_fold, filename + '.npy')
        plot_path = os.path.join(plots_fold, filename + '.npy')
        if not os.path.exists(vertices_path):
            mask = np.load(path)
            if args.toplot:
                from matplotlib import pyplot as plt
                temp = mask[...,0][crop_from:crop_to, crop_from:crop_to] if tocrop else mask[...,0]
                plt.imshow(temp)
                plt.show()
            mask = np.where(mask > 0, 1, 0)
            mask = np.moveaxis(mask, -1, 0)
            if tocrop:
                mask = crop_volume(mask, crop_size=crop_size)
            mask = np.concatenate([mask, mask, mask], axis=0)
            point_cloud = np.zeros((crop_size, crop_size)) if tocrop else np.zeros((256, 256))
            vertices_array = np.zeros((number_points, dim))
            if mask.sum() > 50:
                vol = mcubes.smooth(mask)
                vertices, triangles = mcubes.marching_cubes(vol, 0)
                try:
                    vertices = graipher(vertices, number_points, dim=dim)
                except:
                    print(filename)
                    exit()
                vertices_array = np.array(vertices, dtype=np.int)
                if args.toplot:
                    fig = plt.figure()
                    from mpl_toolkits.mplot3d import Axes3D
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=10)
                    plt.show()
                point_cloud[vertices_array[:,1], vertices_array[:,2]] = 1
                if args.toplot:
                    plt.imshow(point_cloud, cmap='gray')
                    plt.show()

            if to_save=='v' or to_save=='':
                np.save(vertices_path, vertices_array)
            if to_save=='p' or to_save=='':
                np.save(plot_path, point_cloud)
                # mcubes.export_mesh(vertices, triangles, "heart_single_slice.dae", "MyHeart_s")
    print("finish")


def npy2point_datagenerator(mask=None, number_points=300, dim=3, crop_size=112, tocrop=False):
    import mcubes
    mask = np.where(mask > 0, 1, 0)
    mask = np.moveaxis(mask, -1, 0)
    vertices_array = np.zeros((number_points, dim))
    if mask.sum() > 50:
        if tocrop:
            mask = crop_volume(mask, crop_size=crop_size)
        mask = np.concatenate([mask, mask, mask], axis=0)
        # vol = mcubes.smooth(mask)
        vertices, triangles = mcubes.marching_cubes(mask, 0)
        if len(vertices) > 0:
            vertices = graipher(vertices, number_points, dim=dim)
            vertices_array = np.array(vertices, dtype=np.int)
    return vertices_array


def npy2point_datagenerator3d(mask=None, number_points=3600, dim=3):
    import mcubes
    mask = np.where(mask > 0, 1, 0)
    mask = np.moveaxis(mask, -1, 0)
    # vertices_array = np.zeros((number_points, dim))
    # if mask.sum() > 50:
        # vol = mcubes.smooth(mask)
    vertices, triangles = mcubes.marching_cubes(mask, 0)
    vertices = graipher(vertices, number_points, dim=dim)
    vertices_array = np.array(vertices, dtype=np.int)
    return vertices_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fold", help="the data folder", type=str, default='ct_train')
    parser.add_argument("-toplot", help="whether to plot images", action='store_true')
    args = parser.parse_args()
    assert args.fold == 'ct_train' or args.fold == 'ct_val' or args.fold == 'mr_train' or args.fold == 'mr_val'

    npy2point(folder=args.fold, to_save='v', tocrop=False)
    # npy2point_test(folder=args.fold, to_save='v', tocrop=False)
