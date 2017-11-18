import csv
import GoogLeNet.googlenet
import AlexNet.alexnet
import ResNet.resnet
import SquezeNet.squezenet

googlenet = GoogLeNet.googlenet.GoogLeNet()
alexnet = AlexNet.alexnet.AlexNet()
resnet = ResNet.resnet.SquezeNet
squezenet = SquezeNet.squezenet

ofile = open('googlenet_openimage.csv', 'wb')
writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

count = 0
with open('/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/test/images.csv', 'rb') as f:
    reader = csv.reader(f)
    firstline = True
    for row in reader:
        if firstline:  # skip first line
            firstline = False
            continue
        count += 1
        print "count:", count
        if (row[10]):
            first, list = googlenet.run(row[10])
            first, list = alexnet(row[10])
        else:
            continue
        if (first or list):
            writer.writerow(list)
        else:
            continue

ofile.close()