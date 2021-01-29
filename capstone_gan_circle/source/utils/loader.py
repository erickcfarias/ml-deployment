import tensorflow as tf
import SimpleITK as sitk
import random
import numpy as np
from glob import glob


class DataLoader():

    def __init__(self, config):
        self.data_path = config['input_path']
        self.batch_size = config['batch_size']
        self.hu_scale = (config['hu_scale_min'], config['hu_scale_max'])
        n_imgs = len(glob(self.data_path + 'train/*'))
        self.epoch_size = min(config['epoch_size'], n_imgs)
        self.noise_p = config['noise_prob']
        self.noise_mu = config['noise_mu']
        self.noise_sd = config['noise_sd']
        self.blur_p = config['blur_prob']
        self.downscale = config['downscale_rate']
        self.input_type = config['input_type']
        self.input_size = config['input_size']
        self.training = config['training']
        self.scaler = config['scaler']

    def generate_dataset(self, type='train'):
        if type == 'train':
            gen = self.generator(data='train')
        elif type == 'tune':
            gen = self.generator(data='tune')
        elif type == 'test':
            gen = self.generator(data='test')

        dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.float32, tf.float32, tf.float32),
            (tf.TensorShape([None, None, 1]),
             tf.TensorShape([None, None, 1]),
             tf.TensorShape([None, None, 1]))
        )
        if type == 'train':
            df_iterator = iter(
                dataset
                .take(self.epoch_size)
                .batch(int(self.batch_size))
            )
        if type == 'tune':
            df_iterator = iter(
                dataset
                .take(self.epoch_size)
                .batch(int(1))
            )
        elif type == 'test':
            df_iterator = iter(dataset.shuffle(1))

        return df_iterator

    def generator(self, data):

        if data == 'train':
            files = glob(self.data_path + 'train/*')
            random.shuffle(files)
        elif data == 'tune':
            files = glob(self.data_path + 'tune/*')
            random.shuffle(files)
        elif data == 'test':
            files = glob(self.data_path + 'test/*')
            random.shuffle(files)
            self.training = False

        def gen():
            for img_file in files:
                # Load Image
                img = self.image_preprocess(img_file)
                if self.input_type == 'multi':
                    img_hr = self.resize_image(
                        img,
                        (int(img.shape[0]),
                         int(img.shape[1])),
                        type=tf.image.ResizeMethod.BICUBIC
                    )

                elif self.input_type == 'fixed':
                    img_hr = self.resize_image(
                        img,
                        (self.input_size, self.input_size),
                        type=tf.image.ResizeMethod.BICUBIC
                    )

                # Apply Distortions
                img_lr = img_hr
                if self.training:
                    if random.random() <= self.noise_p:
                        sd = random.uniform(self.noise_mu, self.noise_sd)
                        img_lr = self.add_noise(img, mu=self.noise_mu, sd=sd)

                    if random.random() <= self.blur_p:
                        img_lr = self.gaussian_blur(img, 3, 2, 1)

                # Downscale
                img_lr = self.resize_image(
                    img_lr,
                    (int(img.shape[0] * self.downscale),
                     int(img.shape[1] * self.downscale)),
                    type=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                )

                img_lr = self.resize_image(
                    img_lr,
                    (int(img_hr.shape[0]),
                     int(img_hr.shape[1])),
                    type=tf.image.ResizeMethod.BICUBIC
                )

                yield img_lr, img_hr, img
        return gen

    def read_image(self, image_path):
        img = sitk.ReadImage(image_path)
        img = sitk.GetArrayFromImage(img)
        img = self.transform_to_hu(img)

        return img

    def image_preprocess(self, img):
        img = sitk.ReadImage(img)
        img = sitk.GetArrayFromImage(img)
        img = np.squeeze(img)
        img = self.transform_to_hu(img)
        img = self.normalize_hu(img)
        img = tf.expand_dims(img, 2)
        if self.training:
            img = self.augment_image(img)
        img = tf.cast(img, tf.float32)
        return img

    def transform_to_hu(self, image):
        image = image * 1. + (-32768.)
        return image

    def normalize_hu(self, image):
        # Our values currently range from -1024 to around 2000.
        # Anything above 300 is not interesting to us,
        # as these are simply bones with different radiodensity.
        # A commonly used set of thresholds in the LUNA16
        # competition to normalize between are -1000 and 400.
        if self.scaler == 'tanh':
            image = 2. * ((image - self.hu_scale[0]) / (self.hu_scale[1] - self.hu_scale[0])) - 1.
            image[image > 1.] = 1.
            image[image < -1.] = -1.
        elif self.scaler == 'sigm':
            image = (image - self.hu_scale[0]) / (self.hu_scale[1] - self.hu_scale[0])
            image[image > 1.] = 1.
            image[image < 0.] = 0.
        return image

    def random_crop(self, image, size):
        cropped_image = tf.image.random_crop(
            image, size=[size[0], size[1], 1])

        return cropped_image

    def augment_image(self, image):
        # Random flipping
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        return image

    def add_noise(self, image, mu, sd):
        # Adding Gaussian noise
        noise = tf.cast(tf.random.normal(shape=tf.shape(image), mean=mu,
                                         stddev=sd, dtype=tf.double),
                        tf.float32)
        noise_img = tf.add(image, tf.abs(noise))
        return noise_img

    def gaussian_blur(self, img, kernel_size, sigma, n_channels):
        def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
            x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1,
                         dtype=dtype)
            g = tf.math.exp(-(tf.pow(x, 2)
                              / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
            g_norm2d = tf.pow(tf.reduce_sum(g), 2)
            g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
            g_kernel = tf.expand_dims(g_kernel, axis=-1)

            return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)),
                                  axis=-1)

        blur = _gaussian_kernel(kernel_size, sigma, n_channels, img.dtype)
        img = tf.nn.depthwise_conv2d(img[None], blur, [1, 1, 1, 1], 'VALID')

        return img[0]

    def resize_image(self, image, new_size, type=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
        image = tf.image.resize(
            image,
            (new_size[0], new_size[1]),
            method=type,
            preserve_aspect_ratio=False
        )
        return image

    def undo_scaling(self, img, target_max, target_min):
        img = (img - target_min) / (target_max - target_min)
        img = (img * (self.hu_scale[1] - self.hu_scale[0])) + self.hu_scale[0]

        return img

    def convert_to_tensor(self, image):
        new_image = tf.convert_to_tensor(
            image,
            dtype=tf.float32,
            dtype_hint=None,
            name=None
        )
        return new_image

    @tf.function(experimental_relax_shapes=True)
    def postprocess_batch(self, train_tensor, tune_tensor,
                          w, h, c):
        train_tensor_cropped = tf.map_fn(
            lambda x:
            tf.image.random_crop(
                x, size=[w, h, c]
            ),
            elems=train_tensor
        )
        return tf.concat([train_tensor_cropped, tune_tensor], axis=0)
