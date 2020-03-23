import os
import sys
import random
import shutil
"""
raw download: https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz
"""

def label_of_img(img_path):
    return img_path.replace('.png', '_P.png').replace('images', 'labels')

def get_raw_img_path(raw_imgs_path, img_name):
    return os.path.join(raw_imgs_path, img_name)

def get_all_images(raw_imgs_path):
    for _, _, files in os.walk(raw_imgs_path):
        return [get_raw_img_path(raw_imgs_path, f) for f in files if f.endswith('.png')]
    return []

def create_valset(
    val_images,
    target_img_val_path,
    target_lab_val_path
):
    for raw_val_img in val_images:
        raw_val_lab = label_of_img(raw_val_img)
        val_img_path = os.path.join(target_img_val_path, os.path.basename(raw_val_img))
        val_lab_path = label_of_img(val_img_path)
        if not os.path.exists(val_img_path):
            shutil.copy(raw_val_img, val_img_path)
        if not os.path.exists(val_lab_path):
            shutil.copy(raw_val_lab, val_lab_path)

def create_testset(
    test_images,
    target_img_test_path,
    target_lab_test_path
):
    for raw_test_img in test_images:
        raw_test_lab = label_of_img(raw_test_img)
        test_img_path = os.path.join(target_img_test_path, os.path.basename(raw_test_img))
        test_lab_path = label_of_img(test_img_path)
        if not os.path.exists(test_img_path):
            shutil.copy(raw_test_img, test_img_path)
        if not os.path.exists(test_lab_path):
            shutil.copy(raw_test_lab, test_lab_path)

def create_trainset(
    train_images,
    target_img_train_path,
    target_lab_train_path
):
    for raw_train_img in train_images:
        raw_train_lab = label_of_img(raw_train_img)
        train_img_path = os.path.join(target_img_train_path, os.path.basename(raw_train_img))
        train_lab_path = label_of_img(train_img_path)
        if not os.path.exists(train_img_path):
            shutil.copy(raw_train_img, train_img_path)
        if not os.path.exists(train_lab_path):
            shutil.copy(raw_train_lab, train_lab_path)

def split_imgs(val_txt_path, raw_imgs_path, all_images, train_images, val_images, test_images):
    # val
    val_file = open(val_txt_path, "r")
    lines = val_file.read().split('\n')
    for l in lines:
        if len(l) == 0:
            continue
        val_img_path = get_raw_img_path(raw_imgs_path, l)
        all_images.remove(val_img_path)
        val_images.append(val_img_path)
    val_file.close()
    
    # test
    test_ratio = 0.20
    test_count = int(len(all_images) * test_ratio)
    while test_count > 0:
        index = random.randint(0, len(all_images) - 1)
        test_images.append(all_images[index])
        all_images.pop(index)
        test_count -= 1
    
    # train
    train_images += all_images

def refresh_dirs(
    processed_path,
    target_img_train_path,
    target_img_val_path,
    target_img_test_path,
    target_lab_train_path,
    target_lab_val_path,
    target_lab_test_path
):
    shutil.rmtree(processed_path)
    paths = (    
        target_img_train_path,
        target_img_val_path,
        target_img_test_path,
        target_lab_train_path,
        target_lab_val_path,
        target_lab_test_path
    )
    for path in paths:
        os.makedirs(path)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    raw_path = os.path.join(base_dir, 'raw/CamVid')
    raw_imgs_path = os.path.join(raw_path, 'images')
    val_txt_path = os.path.join(raw_path, 'valid.txt')

    processed_path = os.path.join(base_dir, 'processed/CamVid')
    processed_imgs_path = os.path.join(processed_path, 'images')
    processed_labs_path = os.path.join(processed_path, 'labels')


    target_img_train_path = os.path.join(processed_imgs_path, 'train')
    target_img_val_path = os.path.join(processed_imgs_path, 'val')
    target_img_test_path = os.path.join(processed_imgs_path, 'test')
    target_lab_train_path = os.path.join(processed_labs_path, 'train')
    target_lab_val_path = os.path.join(processed_labs_path, 'val')
    target_lab_test_path = os.path.join(processed_labs_path, 'test')

    refresh_dirs(
        processed_path,
        target_img_train_path,
        target_img_val_path,
        target_img_test_path,
        target_lab_train_path,
        target_lab_val_path,
        target_lab_test_path
    )

    all_images = []
    train_images = []
    val_images = []
    test_images = []

    all_images = get_all_images(raw_imgs_path)

    split_imgs(val_txt_path=val_txt_path,
                raw_imgs_path=raw_imgs_path,
                all_images=all_images,
                train_images=train_images,
                val_images=val_images,
                test_images=test_images)

    create_valset(
        val_images,
        target_img_val_path,
        target_lab_val_path
    )
    create_testset(
        test_images,
        target_img_test_path,
        target_lab_test_path
    )
    create_trainset(
        train_images,
        target_img_train_path,
        target_lab_train_path
    )

if __name__ == '__main__':
    main()