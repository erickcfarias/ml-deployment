import tensorflow as tf
import argparse
import yaml
import os
import numpy as np
import json
from model.circle_gan import ganCIRCLE
from utils.loader import DataLoader
from tensorflow.keras.utils import Progbar
from utils.image_tools import calculate_image_similarity

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    # ARGUMENTS FOR SAGEMAKER TRAINING
    parser.add_argument('--model_dir', type=str, default=os.getcwd())
    # parser.add_argument('--sm-model-dir', type=str,
    #                     default=os.environ.get('SM_MODEL_DIR'))
    # parser.add_argument('--train', type=str,
    #                     default=os.environ.get('SM_CHANNEL_TRAINING'))
    # parser.add_argument('--hosts', type=list,
    #                     default=json.loads(os.environ.get('SM_HOSTS')))
    # parser.add_argument('--current-host', type=str,
    #                     default=os.environ.get('SM_CURRENT_HOST'))
    # parser.add_argument('--epochs', type=int,
    #                     default=10)
    # parser.add_argument('--batch_size', type=int,
    #                     default=16)
    # parser.add_argument('--checkpoint_folder', type=str)

    parser.add_argument('-c', '--config_file', default='config/config.yaml',
                        help='Config file for data Preprocessing and GAN training.')
    parser.add_argument('-k', '--key', default=None, help='AWS KEY')
    parser.add_argument('-s', '--secret', default=None, help='AWS secret')

    return parser.parse_known_args()

def parse_config_file(config_file) -> dict:
    """ Read YAML file and do some additional processing.
    """
    config = {}
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    return config


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

    args, unknown = _parse_args()
    config = parse_config_file(args.config_file)

    loader = DataLoader(config)
    circle = ganCIRCLE(config, args.key, args.secret)
    circle.g_lr_hr.save('sr_generator_model')
    
    epochs = config["epochs"]

    iterations = tf.constant(
        (loader.epoch_size
            // (loader.batch_size)) - 2
    )
    tf.print('Each epoch will have %s iterations' % (iterations.numpy()))

    pb = Progbar(tf.cast(iterations * epochs, tf.float32),
                    width=10, verbose=1)

    for epoch in tf.range(epochs):

        g_total_loss_list = []
        d_total_loss_list = []

        train_it = loader.generate_dataset(type='train')
        test_it = loader.generate_dataset(type='test')
        if config['input_type'] == 'multi':
            multi_size_it = loader.generate_dataset(type='tune')

        tf.print("\n epoch {}/{}".format(epoch, epochs))

        for i in tf.range(iterations):

            try:
                train_batch = train_it.get_next()

                if config['input_type'] == 'multi':
                    multi_size_batch = multi_size_it.get_next()
                    w = tf.constant(multi_size_batch[0].shape[1])
                    h = tf.constant(multi_size_batch[0].shape[2])
                    c = tf.constant(1)

                    lr_batch = loader.postprocess_batch(
                        train_batch[0][:-1], multi_size_batch[0], w, h, c
                    )

                    hr_batch = loader.postprocess_batch(
                        train_batch[1][:-1], multi_size_batch[1], w, h, c
                    )

                elif input_type == 'fixed':
                    lr_batch = train_batch[0]
                    hr_batch = train_batch[1]

                d_total_loss = circle.disc_train_step(
                    lr_batch, hr_batch
                )
                d_total_loss_list.append(d_total_loss)

                g_total_loss = circle.gen_train_step(
                    lr_batch, hr_batch
                )
                g_total_loss_list.append(g_total_loss)

                pb.add(1.)

            except tf.python.framework.errors_impl.OutOfRangeError:
                # tf.print("\n OutOfRangeError - Regenerating tf.Datasets")
                train_it = loader.generate_dataset(type='train')
                test_it = loader.generate_dataset(type='test')
                if config['input_type'] == 'multi':
                    multi_size_it = loader.generate_dataset(type='tune')

            except tf.python.framework.errors_impl.InvalidArgumentError:
                # tf.print("\n %s" % (e))
                train_it = loader.generate_dataset(type='train')
                test_it = loader.generate_dataset(type='test')
                if config['input_type'] == 'multi':
                    multi_size_it = loader.generate_dataset(type='tune')

        # Log metrics on tensorboard
        test_batch = test_it.get_next()
        sample_img = circle.g_lr_hr(tf.expand_dims(test_batch[0], 0))

        log_images = tf.concat([
            tf.expand_dims(test_batch[0], 0),
            tf.expand_dims(test_batch[1], 0),
            sample_img
        ], axis=0)

        # Calculate SSIM, PSNR and AMBE
        ssim, psnr, ambe = calculate_image_similarity(
            np.squeeze(sample_img),
            np.squeeze(np.expand_dims(test_batch[1], 0))
        )

        tf.summary.scalar(
            'ssim', data=ssim,
            step=int(circle.checkpointer.step)
        )
        tf.summary.scalar(
            'psnr', data=psnr,
            step=int(circle.checkpointer.step)
        )
        tf.summary.scalar(
            'gen_loss', data=tf.reduce_mean(g_total_loss_list[-10:]),
            step=int(circle.checkpointer.step)
        )
        tf.summary.scalar(
            'disc_loss', data=tf.reduce_mean(d_total_loss_list[-10:]),
            step=int(circle.checkpointer.step)
        )
        tf.summary.image(
            "Image log", log_images, max_outputs=3,
            step=int(circle.checkpointer.step)
        )

        # Checkpoint
        circle.tf_checkpoint()

    #Storing model artifacts
    circle.g_lr_hr.save('sr_generator.h5')
