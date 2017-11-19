import csv
import urllib2
from pathlib2 import Path

import GoogLeNet.googlenet
import AlexNet.alexnet
import ResNet.resnet
import SquezeNet.squezenet


def file_exists(url, is_path):
    if (is_path):
        if Path(url).is_file():
            return True
        else:
            return False
    else:
        try:
            f = urllib2.urlopen(urllib2.Request(url))
            deadLinkFound = True
        except:
            deadLinkFound = False
        return deadLinkFound


def calc_results(class_name, first, list):
    if class_name in first:
        top1 = 1
    else:
        top1 = 0
    count = 1
    for item in list:
        if class_name in item[1]:
            top5 = count
            break
        else:
            top5 = 0
        count += 1

    return top1, top5


########
# MAIN #
########
if __name__ == "__main__":

    class_names = ['car', 'airplane', 'truck', 'dog', 'cat', 'horse', 'ship', 'bird', 'bicycle', 'cow']

    googlenet = GoogLeNet.googlenet.GoogLeNet()
    alexnet = AlexNet.alexnet.AlexNet()
    resnet = ResNet.resnet.ResNet()
    squezenet = SquezeNet.squezenet.SquezeNet()

    for class_name in class_names:

        ofile = open('results_openimage_' + class_name + '.csv', 'wb')
        writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        image_path = '/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/images/'

        count = 0
        with open('/home/samuel/PycharmProjects/convnet_transfer_learning/openimage_class_' + class_name + '.csv',
                  'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                count += 1
                print "count:", count
                if (row[1]):
                    image_file = image_path + row[1]
                    if (file_exists(image_file, True)):
                        googlenet_first, googlenet_list = googlenet.run(image_file)
                        alexnet_first, alexnet_list = alexnet.run(image_file)
                        resnet_first, resnet_list = resnet.run(image_file)
                        squezenet_first, squezenet_list = squezenet.run(image_file)

                        googlenet_top1, googlenet_top5 = calc_results(class_name, googlenet_first, googlenet_list)
                        alexnet_top1, alexnet_top5 = calc_results(class_name, alexnet_first, alexnet_list)
                        resnet_top1, resnet_top5 = calc_results(class_name, resnet_first, resnet_list)
                        squezenet_top1, squezenet_top5 = calc_results(class_name, squezenet_first, squezenet_list)

                        row_result = [row[0], row[1], row[3],
                                      googlenet_list, googlenet_top1, googlenet_top5,
                                      alexnet_list, alexnet_top1, alexnet_top5,
                                      resnet_list, resnet_top1, resnet_top5,
                                      squezenet_list, squezenet_top1, squezenet_top5]

                        writer.writerow(row_result)
                else:
                    continue
                    # if (first or list):
                    #     writer.writerow(list)
                    # else:
                    #     continue

        ofile.close()
