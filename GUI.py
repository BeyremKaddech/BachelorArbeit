import matplotlib
matplotlib.use('Agg')
import os, sys
sys.path.append(os.getcwd())


import functools 
from tkinter import Tk, Label, Button, LEFT, RIGHT
from functools import partial
import numpy as np
import tensorflow as tf
import sklearn.datasets


import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.ops.layernorm



DATA_DIR = '/var/tmp/ga94pak/Code2'
DIM = 64
BATCH_SIZE = 64
OUTPUT_DIM = 64*64*3


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output




# ! Generators

def GoodGenerator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu):
    print(n_samples)
    if noise is None:
        noise = tf.random_normal([int(n_samples), 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up')

    output = Normalize('Generator.OutputN', [0,2,3], output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])





class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("Van Gogh image generator!")

        self.label = Label(master, text="Please click on one of the buttons below to generate a painting")
        self.label.pack(side=LEFT)
        self.bw_button = Button(master, text="Generate a Black and White painting", command=partial(self.load_generate, "/BW"))
        self.bw_button.pack(side=RIGHT)
        self.face_button = Button(master, text="Generate a portrait painting ", command=partial(self.load_generate, "/faces"))
        self.face_button.pack()
        self.abstract_button = Button(master, text="Generate an abstract Van Gogh style painting",
                                      command=partial(self.load_generate, "/5x5mosa"))
        self.abstract_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def load_generate(self, model_dir):

        def generate_image(iteration):
            samples = session.run(all_fixed_noise_samples)
            print(samples.shape)
            samples = ((samples + 1.) * (255.99 / 2)).astype('int32')
            lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), 'Generated{}.png'.format(model_dir))


        saver = tf.train.Saver()
        saver.restore(session, DATA_DIR + model_dir +"/model.ckpt")
        print("Model restored.")
        iteration = 0
        n_samples = BATCH_SIZE
        fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
        all_fixed_noise_samples = []
        all_fixed_noise_samples.append(GoodGenerator(n_samples, noise=fixed_noise[0:n_samples]))
        if tf.__version__.startswith('1.'):
            all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
        else:
            all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)
        print(all_fixed_noise_samples.shape)

        print('Generating images...')
        generate_image(iteration)





Generator = GoodGenerator(BATCH_SIZE)


session = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))

root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()




