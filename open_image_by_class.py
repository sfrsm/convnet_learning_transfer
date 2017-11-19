import csv

class_names = ['car', 'airplane', 'truck', 'dog', 'cat', 'horse', 'ship', 'bird', 'bicycle', 'cow']

images_file = open('/home/samuel/PycharmProjects/convnet_transfer_learning/openimage_class-final.csv', 'rb')
images_reader = csv.reader(images_file)

for class_name in class_names:
    ofile = open('openimage_class_' + class_name + '.csv', 'wb')
    writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

    images_file.seek(0)

    # image_id image_file class_id class_name
    images_id = set()

    row_dict = {}

    for row_image in images_reader:
        if (row_image[1]):
            if (class_name in row_image[3].lower()):
                images_id.add(row_image[0])
                row_dict[row_image[0]] = row_image

    for id in images_id:
        writer.writerow(row_dict[id])

    ofile.close()

print "Fim!"