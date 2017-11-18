import csv

ofile = open('openimage_class.csv', 'a')
writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

images_file = open('/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/test/images.csv', 'rb')
annotation_file = open('/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/test/annotations-human.csv', 'rb')
class_file = open('/home/samuel/PycharmProjects/convnet_transfer_learning/open_image/class-descriptions.csv', 'rb')

images_reader = csv.reader(images_file)
class_reader = csv.reader(class_file)
annotation_reader = csv.reader(annotation_file)

first_line = True
restart = True
for row_image in images_reader:
    if (first_line):
        first_line = False
        continue
    image_id = row_image[0]
    if (restart and (image_id <> 'ee5a91b811a8ce3e')):
        continue
    restart = False
    image_file = row_image[10].split('/')[-1]
    annotation_file.seek(0)
    print "Image: ", image_file
    for row_annotation in annotation_reader:
        if image_id in row_annotation:
            if row_annotation[3] == '1':
                label = row_annotation[2]
                class_file.seek(0)
                for row_class in class_reader:
                    if label in row_class:
                        row = [image_id, image_file, label, row_class[1]]
                        writer.writerow(row)
ofile.close()
print "Fim!"

