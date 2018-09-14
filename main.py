from __future__ import print_function

import numpy as np
import face_recognition
import argparse
import cv2
import os
import pickle
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from datetime import datetime
from module import *
from utils import *
from model import cyclegan
now = datetime.now()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='face2zebra', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')

args2 = parser.parse_args()
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,help="path to output video file")
ap.add_argument("-p", "--picamera", type=int, default=-1, help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=50, help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG", help="codec of output video")
ap.add_argument('--with_draw', help='do draw?', default='True')
args = vars(ap.parse_args())
##
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None

mode = 'knn'
video = 'v'

net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')
knn_clf = pickle.load(open('./models/fr_knn.pkl', 'rb'))
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
def preprocess(img):
    ### analysis
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(1):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_img.mean() < 130:
            img = adjust_gamma(img, 1.5)
        else:
            break
    return img

if video == 'v':
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args2)
        model.loadModel()
        vc = cv2.VideoCapture('./data/video3.mp4')
        length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        print ('length :', length)
        if args["with_draw"] == 'True':
            cv2.namedWindow('show', 0)
        dir_name = str(now.year)+ "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.minute) + "_" + str(now.second) #+ "_" + str(idx)
        os.makedirs(dir_name)
        print(dir_name)
        for idx in range(length):
            img_bgr = vc.read()[1]
            if img_bgr is None:
                break
            start = cv2.getTickCount()
        ### preprocess
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_bgr_ori = img_bgr.copy()
            print(img_bgr_ori.shape)
            img_bgr = preprocess(img_bgr)

        ### detection
            border = (img_bgr.shape[1] - img_bgr.shape[0])//2
            img_bgr = cv2.copyMakeBorder(img_bgr,
                                        border, # top
                                        border, # bottom
                                        0, # left
                                        0, # right
                                        cv2.BORDER_CONSTANT,
                                        value=(0,0,0))

            (h, w) = img_bgr.shape[:2]
        
            blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

        ### bbox
            list_bboxes = []
            list_confidence = []
        # list_dlib_rect = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.6:
                        continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (l, t, r, b) = box.astype("int") # l t r b

                original_vertical_length = b-t
                t = int(t + (original_vertical_length)*0.15) - border
                b = int(b - (original_vertical_length)*0.05) - border

                margin = ((b-t) - (r-l))//2
                l = l - margin if (b-t-r+l)%2 == 0 else l - margin - 1
                r = r + margin
                if(l >w) l=w
                if(t <0) t=0
                if(r <0) r=0
                if(b >h) b=h

                
                refined_box = [t,r,b,l]
                list_bboxes.append(refined_box)
                list_confidence.append(confidence)
        ### facenet
            if(len(list_bboxes)>0) :
                face_encodings = face_recognition.face_encodings(img_rgb, list_bboxes)
        
                if mode == "knn":
                    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
                    is_recognized = [closest_distances[0][i][0] <= 0.4 for i in range(len(list_bboxes))]
                    list_reconized_face = [(pred, loc, conf) if rec else ("unknown", loc, conf) for pred, loc, rec, conf in zip(knn_clf.predict(face_encodings), list_bboxes, is_recognized, list_confidence)]
        
                elif mode == "svm":
                    predictions = knn_clf.predict_proba(face_encodings)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    is_recognized = [ best_class_indices == 1 for i in range(len(list_bboxes))]
                    list_reconized_face = [(pred, loc, conf) if rec else ("unknown", loc, conf) for pred, loc, rec, conf in zip(knn_clf.predict(face_encodings), list_bboxes, best_class_indices, list_confidence)]
                time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
                print ('%d, elapsed time: %.3fms'%(idx,time))
                
                img_bgr_blur = img_bgr_ori.copy()
                
                for name, bbox, conf in list_reconized_face:
                    t,r,b,l = bbox
                    if name == 'unknown':
                        file_name = "0001.jpg"
                        
                        crop_img = img_bgr_blur[t:b, l:r]
                        crop_img2 = cv2.resize(crop_img,(256,256))
                        cv2.imwrite(os.path.join(dir_name , file_name) ,crop_img2)
                        height, width = crop_img.shape[:2]
                        
                        os.system("mv -f " +dir_name +"/* ./data/face2zebra/testA")         
                        model.extract()
                        gan_img  = cv2.imread(os.path.join('./test' , 'AtoB_' +file_name) , 1)
                        
                        gan_img2 = cv2.resize(gan_img,(height,width))
                        img_bgr_blur[t:b, l:r] = gan_img2

                if args["with_draw"] == 'True':
                    source_img = Image.fromarray(img_bgr_blur)
                    draw = ImageDraw.Draw(source_img)
                    for name, bbox, confidence in list_reconized_face:
                        t,r,b,l = bbox
                        font_size = int((r-l)/img_bgr_ori.shape[1]*200)

                        # draw.rectangle(((l,t),(r,b)), outline=(0,255,128))

                        # draw.rectangle(((l,t-font_size-2),(r,t+2)), fill=(0,255,128))
                        # draw.text((l, t - font_size), name, font=ImageFont.truetype('./BMDOHYEON_TTF.TTF', font_size), fill=(0,0,0,0))
                    show = np.asarray(source_img)
                    if idx == 0:
                        writer = cv2.VideoWriter('video_3.avi', fourcc, args["fps"], (640,360), True)
                    writer.write(show)
            else :
                source_img = Image.fromarray(img_bgr_ori)
                draw = ImageDraw.Draw(source_img)
                show = np.asarray(source_img)
                if idx == 0:
                    writer = cv2.VideoWriter('video_3.avi', fourcc, args["fps"], (640,360), True)
                writer.write(show)
        os.system("rmdir "+dir_name)
writer.release()

