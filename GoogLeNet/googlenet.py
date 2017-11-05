import numpy as np
import caffe

####################
# montagem da rede #
####################

caffe.set_mode_cpu()

base_dir = '/home/samuel/caffe/'
model_dir = 'models/bvlc_googlenet/'

model_def = base_dir + model_dir + 'deploy.prototxt'
model_weights = base_dir + model_dir + 'bvlc_googlenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

#############################################
# configuracao para transformacao da imagem #
#############################################
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(base_dir + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
# net.blobs['data'].reshape(50,        # batch size
#                           3,         # 3-channel (BGR) images
#                           227, 227)  # image size is 227x227

#####################
# carregando imagem #
#####################
# image = caffe.io.load_image(base_dir + 'examples/images/cat.jpg')
image = caffe.io.load_image('http://farm4.staticflickr.com/3177/2740694473_cfd1f8140a_z.jpg')
transformed_image = transformer.preprocess('data', image)
# plt.imshow(image)

#################
# classificacao #
#################
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()

#################################
# carregando labels da ImageNet #
#################################
# load ImageNet labels
labels_file = base_dir + 'data/ilsvrc12/synset_words.txt'

labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:', zip(output_prob[top_inds], labels[top_inds])