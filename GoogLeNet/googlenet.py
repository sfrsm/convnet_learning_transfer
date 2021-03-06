import numpy as np
import caffe
import tensorflow as tf
import urllib2

#bvlc_googlenet model read.me

class GoogLeNet:
    def __init__(self):
        caffe.set_mode_cpu()

        self.base_dir = '/home/samuel/caffe/'
        model_dir = 'models/bvlc_googlenet/'

        model_def = self.base_dir + model_dir + 'deploy.prototxt'
        model_weights = self.base_dir + model_dir + 'bvlc_googlenet.caffemodel'

        self.net = caffe.Net(model_def,  # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)

        #############################################
        # configuracao para transformacao da imagem #
        #############################################
        # load the mean ImageNet image (as distributed with Caffe) for subtraction
        mu = np.load(self.base_dir + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
        mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        # print 'mean-subtracted values:', zip('BGR', mu)

        # create transformer for the input called 'data'
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        self.transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        self.transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
        self.transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    def exists(self, url, is_path):
        if (is_path):
            if Path(url).is_file():
                return False
            else:
                return True
        else:
            try:
                f = urllib2.urlopen(urllib2.Request(url))
                deadLinkFound = False
            except:
                deadLinkFound = True
            return deadLinkFound


    def run(self, imagefile):
        try:
            # set the size of the input (we can skip this if we're happy
            #  with the default; we can also change it later, e.g., for different batch sizes)
            # self.net.blobs['data'].reshape(50,  # batch size
            #                                3,  # 3-channel (BGR) images
            #                                224, 224)  # image size is 224x224

            #####################
            # carregando imagem #
            #####################
            # image = caffe.io.load_image(base_dir + 'examples/images/cat.jpg')
            image = caffe.io.load_image(imagefile)
            transformed_image = self.transformer.preprocess('data', image)
            # plt.imshow(image)

            #################
            # classificacao #
            #################
            # copy the image data into the memory allocated for the net
            self.net.blobs['data'].data[...] = transformed_image

            ### perform classification
            output = self.net.forward()

            output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

            # print 'predicted class is:', output_prob.argmax()

            #################################
            # carregando labels da ImageNet #
            #################################
            # load ImageNet labels
            labels_file = self.base_dir + 'data/ilsvrc12/synset_words.txt'

            labels = np.loadtxt(labels_file, str, delimiter='\t')

            # print 'output label:', labels[output_prob.argmax()]

            # sort top five predictions from softmax output
            top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

            # print 'probabilities and labels:', zip(output_prob[top_inds], labels[top_inds])

            return labels[output_prob.argmax()], zip(output_prob[top_inds], labels[top_inds])
        except:
            return -1, -1

    def LoadLabelMap(self):
      """Load index->mid and mid->display name maps.
    
      Args:
        labelmap_path: path to the file with the list of mids, describing
            predictions.
        dict_path: path to the dict.csv that translates from mids to display names.
      Returns:
        labelmap: an index to mid list
        label_dict: mid to display name dictionary
      """
      labelmap_path = '/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/classes-trainable.txt'
      dict_path = '/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/class-descriptions.csv'

      labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path)]

      label_dict = {}
      for line in tf.gfile.GFile(dict_path):
        words = [word.strip(' "\n') for word in line.split(',', 1)]
        label_dict[words[0]] = words[1]

      return labelmap, label_dict


if __name__ == "__main__":
    googlenet = GoogLeNet()
    # googlenet.run('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRjOqKI0kZG7nIV2w7AFRWfPUGiqeM0J26TbCp8irR1jZiNG556')
    #first, list = googlenet.run('https://vetstreet.brightspotcdn.com/dims4/default/a1a90c7/2147483647/thumbnail/180x180/quality/90/?url=https%3A%2F%2Fvetstreet-brightspot.s3.amazonaws.com%2F0d%2Ff2e4c0b3a611e092fe0050568d634f%2Ffile%2Fhub-cats-senior.jpg')
    # first, list = googlenet.run('https://c1.staticflickr.com/3/2934/14439122755_f967f0bae5_z.jpg')
    first, list = googlenet.run('/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/images/4399852181_a3d116f71f_z.jpg')
    print "first:", first
    print "list:" , list