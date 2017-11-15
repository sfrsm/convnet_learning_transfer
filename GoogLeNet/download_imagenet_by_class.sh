#!/bin/bash

imagenet_dir="imagenet_utils"

echo "Get the imagenet_utils repository"

if [ ! -d $imagenet_dir ]; then
    git clone https://github.com/tzutalin/ImageNet_Utils.git imagenet_utils
else
    echo "imagenet_utils already exists!"
fi


cd $imagenet_dir

echo "Downloading images: $1"
class_name=$1
class_wnid="n09618957"
./downloadutils.py --downloadImages --wnid $class_wnid
mkdir ../data/$class_name
mv $class_wnid/"$class_wnid"_urlimages ../data/$class_name
echo "find images under ../data/$class_name"