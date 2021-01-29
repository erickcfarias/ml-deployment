import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import logging
from glob import glob
import os
import re
from PIL import Image
import subprocess


class DeepLesionPreprocessor:

    def __init__(self, config: dict):
        self.logger = self._get_logger()
        self.input_path = config['input_path']
        self.output_path = config['output_path']
        self.urls = config['data_urls']
        self.download = config['download']
        self.train = config['train']
        self.test = config['test']
        self.delete_raw = config['delete_raw']
        self.input_size = int(config['input_size'])
        self.crop_size = int(config['crop_size'])
        self.multi_size = config['multi_size_input']
        self.patch_constraints = (int(self.crop_size/2), int(self.crop_size))
        self.downscale_rate = self.input_size / self.crop_size
        self.logger.info(
            'Preprocessor loaded.')

    def _get_logger(self):
        logging.basicConfig(
            filename='preprocessing.log',
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        return logging.getLogger(__name__)

    def _check_dirs(self):
        if not os.path.isdir(self.input_path):
            os.mkdir(self.input_path)
            self.logger.info('Folder {} created.'.format(self.input_path))

        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
            self.logger.info('Folder {} created.'.format(self.output_path))

        if not os.path.isdir(self.output_path + 'train'):
            os.mkdir(self.output_path + 'train')
            self.logger.info(
                'Folder {}train created.'.format(self.output_path))

        if not os.path.isdir(self.output_path + 'test'):
            os.mkdir(self.output_path + 'test')
            self.logger.info('Folder {}test created.'.format(self.output_path))

        if not os.path.isdir(self.output_path + 'tune'):
            os.mkdir(self.output_path + 'tune')
            self.logger.info('Folder {}tune created.'.format(self.output_path))

        self.logger.info('Directories created.')

    def _download_data(self, idx, url):
        self.logger.info(
            'Started download file {}/{}.'.format(idx+1, len(self.urls)))

        if not os.path.isdir('download/'):
            os.mkdir('download/')

        # download file
        bashCommand = 'wget {} -O ./download/file.zip'.format(url)
        subprocess.run(
            bashCommand.split(),
            stdout=subprocess.PIPE,
            check=True
        )

        # unzip file
        try:
            bashCommand = 'unzip ./download/file.zip -d ./download/'
            subprocess.run(
                bashCommand.split(),
                stdout=subprocess.PIPE,
                check=True
            )
        except Exception:
            self.logger.warning("Out of disk space. Proceeding with images downloaded so far.")

        # rm file
        bashCommand = 'rm ./download/file.zip'
        subprocess.run(
            bashCommand.split(),
            stdout=subprocess.PIPE,
            check=True
        )

        # move and rename images
        self.logger.info(
            'Renaming images from file {}/{}.'.format(idx+1, len(self.urls)))
        images = glob('download/*/*/*')
        for img in images:
            try:
                x = re.split(r'/', string=img)
                new_path = '{}/{}_{}'.format(self.input_path, x[2], x[3]).\
                    replace("//", "/")
                os.rename(img, new_path)
            except Exception as e:
                self.logger.exception(e)
                continue

    def _prepare_training(self):
        """ for each patient:
                - list all image slices belonging to patient
                    for i in range(20):
                        - select a slice
                        - perform a random crop
                        - validate if patch has less than 50% of air
                        - save patch
        """
        self.logger.info(
            'Started generating random patches from raw image files.')
        self.files = glob(self.input_path + '*')
        self.file_names = [re.search(pattern=r'\d.+\d.png', string=i)[0]
                           for i in self.files]
        self.file_idxs = list(
            set([
                re.sub(pattern=r'_\d{3}.png', repl='', string=f)
                for f in self.file_names
            ])
        )

        for idx in self.file_idxs:
            n_files = len(os.listdir("preprocessed_data/train"))
            print(f"{n_files} train files generated")
            files = [f for f in self.files if idx in f]

            counter = 0
            stop_count = 100
            while True:
                try:
                    rand_img = np.random.choice(files)
                    file_name = re.search(
                        pattern=r'\d.+\d.png', string=rand_img)[0]

                    img = sitk.ReadImage(rand_img)
                    img = sitk.GetArrayFromImage(img)
                    img = tf.expand_dims(img, 2)

                    patch = \
                        tf.image.random_crop(
                            img, (self.crop_size, self.crop_size, 1)
                        )

                    patch = tf.cast(tf.image.resize(
                        patch,
                        (self.input_size, self.input_size),
                        method=tf.image.ResizeMethod.BICUBIC,
                        preserve_aspect_ratio=True
                    ), tf.float32)

                    patch = tf.squeeze(patch).numpy()

                    if self._validate_img_air_proportion(patch, 0.5):

                        patch = patch.astype(np.uint16)
                        Image.fromarray(patch).save(
                            self.output_path + 'train/' + file_name
                        )
                        counter += 1
                    stop_count -= 1
                    if (stop_count <= 0) or (counter > 20):
                        break

                except Exception as e:
                    self.logger.exception(e)
                    continue

    def _prepare_multi_sizing_input(self):
        """ for each patient:
                - list all image slices belonging to patient
                    for i in range(20):
                        - select a slice
                        - perform a random crop
                        - validate if patch has less than 50% of air
                        - save patch
        """
        self.logger.info(
            'Started generating random patches from raw image files.')
        self.files = glob(self.input_path + '*')
        self.file_names = [re.search(pattern=r'\d.+\d.png', string=i)[0]
                           for i in self.files]
        self.file_idxs = list(
            set([
                re.sub(pattern=r'_\d{3}.png', repl='', string=f)
                for f in self.file_names
            ])
        )

        for idx in self.file_idxs:

            files = [f for f in self.files if idx in f]

            stop_count = 100
            counter = 0
            while True:
                try:
                    rand_img = np.random.choice(files)
                    file_name = re.search(
                        pattern=r'\d.+\d.png', string=rand_img)[0]

                    img = sitk.ReadImage(rand_img)
                    img = sitk.GetArrayFromImage(img)
                    img = tf.expand_dims(img, 2)

                    x = random.randint(
                        self.patch_constraints[0], self.patch_constraints[1]
                    )
                    x = x if x % 2 == 0 else x + 1
                    y = random.randint(
                        self.patch_constraints[0], self.patch_constraints[1]
                    )
                    y = y if y % 2 == 0 else y + 1
                    patch = \
                        tf.image.random_crop(
                            img, (x, y, 1)
                        )
                    patch_x = int(patch.shape[0] * 0.25)
                    while True:
                        if patch_x % 8 == 0:
                            break
                        else:
                            patch_x = np.round(patch_x + 1)

                    patch_y = int(patch.shape[1] * 0.25)
                    while True:
                        if patch_y % 8 == 0:
                            break
                        else:
                            patch_y = np.round(patch_y + 1)

                    patch = tf.cast(tf.image.resize(
                        patch,
                        (patch_x,
                         patch_y),
                        method=tf.image.ResizeMethod.BICUBIC,
                        preserve_aspect_ratio=False
                    ), tf.float32)

                    patch = tf.squeeze(patch).numpy()

                    if self._validate_img_air_proportion(patch, 0.5):
                        
                        patch = patch.astype(np.uint16)
                        Image.fromarray(patch).save(
                            self.output_path + 'tune/' + file_name
                        )
                        counter += 1
                    stop_count -= 1
                    if (stop_count <= 0) or (counter > 20):
                        break

                except Exception as e:
                    self.logger.exception(e)
                    continue

    def _prepare_testing(self):
        """ Select randomly x images from fine tuning folder and move them
        """
        images = glob(self.output_path + 'train/*')
        images = np.random.choice(images, size=50, replace=False)

        for img in images:
            x = re.split(r'/', string=img)
            new_path = '{}{}'.format(self.output_path + "test/", x[2])
            os.rename(img, new_path)

    def _delete_folder(self, folder_path: str):
        bashCommand = "rm -r {}".format(folder_path)
        subprocess.run(
            bashCommand.split(),
            stdout=subprocess.PIPE,
            check=True
        )

    def _validate_img_air_proportion(self, img: np.array, proportion_threshold: float) -> bool:

        img = img * 1. + (-32768.)
        mask = img <= -100
        prop_air = np.sum(mask) / (mask.shape[0] * mask.shape[1])

        if prop_air <= proportion_threshold:
            return True
        else:
            return False

    def run(self):
        for idx, url in enumerate(self.urls):
            self._check_dirs()
            if self.download:
                self._download_data(idx, url)
                self._delete_folder('download/')
            if self.train:
                self._prepare_training()
                self.logger.info(
                    'Finished generating random patches for training.')
            if self.multi_size:
                self._prepare_multi_sizing_input()
                self.logger.info(
                    'Finished generating random size patches for multi sized input.')
            if self.test:
                self._prepare_testing()
                self.logger.info(
                    'Finished generating lesion centered patches for test images.')
            if self.delete_raw:
                self._delete_folder(self.input_path)

            # Check amount of images generated:
            if len(os.listdir("preprocessed_data/train")) > 5000:
                break
