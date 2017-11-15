import csv
import urllib


count = 0
with open('/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/test/images.csv', 'rb') as f:
    reader = csv.reader(f)
    firstline = True
    for row in reader:
        if firstline:  # skip first line
            firstline = False
            continue
        count += 1
        if (count < 41051):
            continue
        #print "count:", count
        #print "teste: ", '/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/images/' + str(row[10].split('/')[-1])
        if (row[10]):
            urllib.urlretrieve(row[10], '/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/images/' + str(row[10].split('/')[-1]))

