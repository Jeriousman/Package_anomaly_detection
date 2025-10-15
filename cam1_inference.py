import os
import cv2
import time
import json
import datetime
import requests

import torch
import pyjds
import numpy as np
import traceback
import logging.handlers

from multiprocessing import shared_memory
from utils.crop_image import crop
from utils.db_insert import db_insert
from utils.save_image import save_image
from utils.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from utils.mask_adjustment import mask_adjustment

time.sleep(10)

################################################################
#                            Config                            #
################################################################
with open('C:\\Users\\SDT-SHPNP\\Workspace\\ad_inference\\configs\\config.json', 'r') as f:
    info = json.load(f)

info['pid_cam1'] = os.getpid()

UNIT = info['unit']
GPU_ID = info['gpu_id']
CAM_NUM = 1
OBJ_NAME = info['obj_name_cam1']
BASE_MODEL_NAME = info['base_model_name']
CHECKPOINT_PATH = info['checkpoint_path']
RUN_NAME = BASE_MODEL_NAME + "_" + OBJ_NAME + '_'
IMG_HEIGHT, IMG_WIDTH = info['image_size']
MASK_CONFIG = info['mask_config']
SLACK_URL = info['slack_url']
DUMMY_IMG = info['dummy_img_for_model_init']
MEMORY_ADDRESS = info['shared_memory_path']

with open('C:\\Users\\SDT-SHPNP\\Workspace\\ad_inference\\configs\\config.json', 'w') as f:
    json.dump(info, f, indent=4)
    
# THRESHOLD = 0.01244497
THRESHOLD = 0.15
MASK, BBOX = crop(MASK_CONFIG, 1)

Vsq = 1

################################################################
#                        Save Log File                         #
################################################################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_max_size = 1024000
log_file_count = 3
log_fileHandler = logging.handlers.RotatingFileHandler(
    filename=f"C:\\Users\\SDT-SHPNP\\Workspace\\ad_inference\\logs\\shpnp_ad_{OBJ_NAME}.log",
    maxBytes=log_max_size,
    backupCount=log_file_count,
    mode='a')

log_fileHandler.setFormatter(formatter)
logger.addHandler(log_fileHandler)


################################################################
#                         status check                         #
################################################################
def send_message_to_slack(message):
    data = {"text": message}
    req = requests.post(
        url=SLACK_URL,
        data=json.dumps(data)
    )


################################################################
#                         Define Model                         #
################################################################
class ad_models:
    def __init__(self):
        self.recon_model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        self.recon_model.cuda()
        self.recon_model.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_PATH, RUN_NAME + ".pckl"), map_location='cuda:0'))
        self.recon_model.eval()

        self.seg_model = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        self.seg_model.cuda()
        self.seg_model.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_PATH, RUN_NAME + "_seg.pckl"), map_location='cuda:0'))
        self.seg_model.eval()

        ########################################################
        #                  Connect to Camera                   #
        ########################################################
        self.camera = pyjds.DeviceFinder().find()
        self.test_connect = pyjds.DeviceFactory.create(self.camera[2])
        self.test_connect.connect()

        self.streams = self.test_connect.create_and_open_streams()
        # self.test_connect.acquisition_stop()


################################################################
#                     Image Preprocessing                      #
################################################################
def image_preprocessing(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('./template/b1.jpg', 0)
    template_top_left = (896, 701)

    # 템플릿 매칭으로 글자 찾기
    find_template = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)

    # 결과값 꺼내기
    _, _, _, max_loc = cv2.minMaxLoc(find_template)

    # 이미지 위치 보정
    shift_x, shift_y = [a-b for a, b in zip(template_top_left, max_loc)]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    m, (xmin, xmax, ymin, ymax) = MASK, BBOX
    m = m.astype(np.uint8)

    result = cv2.bitwise_and(shifted_img, shifted_img, mask=m)
    # result = cv2.bitwise_and(image, image, mask=m)
    cropped_image = cv2.rotate(result[ymin:ymax, xmin:xmax, :], cv2.ROTATE_180)

    img = cv2.resize(cropped_image.copy(), dsize=(IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.array(img).reshape((img.shape[0], img.shape[1], 3)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))

    return cropped_image, img


################################################################
#                          On Memory                           #
################################################################
def init_state(models):
    _, dummy_image = image_preprocessing(cv2.imread(DUMMY_IMG, cv2.IMREAD_COLOR)) # template match 

    with torch.no_grad():
        gray_batch = torch.Tensor(dummy_image).unsqueeze(0).cuda()
        gray_rec = models.recon_model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = models.seg_model(joined_in)


################################################################
#                          Inference                           #
################################################################
def inference(RGB):
    global models, Vsq 

    cropped_image, image = image_preprocessing(RGB.copy())

    with torch.no_grad():
        ####################################################
        #            Reconstructive Model Preds            #
        ####################################################
        gray_batch = torch.Tensor(image).unsqueeze(0).cuda()
        gray_rec = models.recon_model(gray_batch) # 정상 이미지로 변환
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        ####################################################
        #            Discriminative Model Preds            #
        ####################################################
        out_mask = models.seg_model(joined_in) # 원본 이미지와 정상이미지 사이의 차이를 계산 
        out_mask_sm = torch.softmax(out_mask, dim=1)

        # xmin, xmax, ymin, ymax = BBOX
        # orig_mask = cv2.resize(MASK[ymin:ymax, xmin:xmax], dsize=(512, 256))

        # orig_image = out_mask_sm[:, 1:, :, :][0][0].detach().cpu().numpy() * 255
        # new_image = mask_adjustment(orig_mask, orig_image)

        # out_mask_cv = new_image
        out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        # out_mask_averaged = torch.nn.functional.avg_pool2d(torch.Tensor(new_image).unsqueeze(0).unsqueeze(0), 21, stride=1,
        #                                                      padding=21 // 2).cpu().detach().numpy()

        image_score = np.max(out_mask_averaged)

        ####################################################
        #      Save Model Pred Mask & Original Image       #
        ####################################################
        today = datetime.date.today()
        nas_root_dir = "/volume1/sdt-shpnp-storage-1/unit1_image_bucket"

        if  round(float(image_score), 8) >= THRESHOLD:  # In Error
            img_name = save_image(OBJ_NAME, RGB.copy(), cropped_image, out_mask_cv, 'F')

            image_path = f"{nas_root_dir}/abnormal/{OBJ_NAME}/{str(today)}/image/{img_name}.jpg"
            mask_path = f"{nas_root_dir}/abnormal/{OBJ_NAME}/{str(today)}/mask/{img_name}_mask.jpg"

            db_insert(UNIT, Vsq, "F", CAM_NUM, image_score, image_path, mask_path)
            logger.info(f"name: {OBJ_NAME}/{img_name}.jpg | score: {image_score:.8f} | Abnormal")
        else:
            img_name = save_image(OBJ_NAME, RGB.copy(), cropped_image, out_mask_cv, 'T')

            image_path = f"{nas_root_dir}/normal/{OBJ_NAME}/{str(today)}/image/{img_name}.jpg"
            mask_path = f"{nas_root_dir}/normal/{OBJ_NAME}/{str(today)}/mask/{img_name}_mask.jpg"

            db_insert(UNIT, Vsq, "T", CAM_NUM, image_score, image_path, mask_path)
            logger.info(f"name: {OBJ_NAME}/{img_name}.jpg | score: {image_score:.8f} |   Normal")

        Vsq += 1


if __name__ == "__main__":
    logger.info('initialize')

    # 공유 메모리 주소 접근
    with open(MEMORY_ADDRESS,"r") as f:
        addr = f.read()
    logger.info(f"address: {addr}")
    
    try:
        shm = shared_memory.SharedMemory(name=addr, create=True, size=1)
        print("공유 메모리를 새로 생성했습니다.")
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=addr)
        print("이미 존재하는 공유 메모리에 연결했습니다.")

    models = ad_models()
    init_state(models)  # on memory
    logger.info(f"{OBJ_NAME} Model Load Done. Model name: {RUN_NAME}")
    logger.info(f"---------- {OBJ_NAME} Threshold: {THRESHOLD} ----------")

    prev_stat = False
    cur_stat = False

    while True:
        try:
            # for test
            # test_image = cv2.imread(f'C:\\Users\\SDT-SHPNP\\Workspace\\ad_inference\\test\\{OBJ_NAME}_brown.jpg')
            # inference(test_image)
            # time.sleep(1)
            # continue

            ######################################################
            #                    Read Serial                     #
            ######################################################
            # with open("./ser_value.txt", "rb") as f:
            #     ser_value = f.read()

            ser_value = shm.buf[:1].tobytes()
            print(ser_value, datetime.datetime.now(), end='\r')

            ######################################################
            #                      Inference                     #
            ######################################################
            if ser_value == b'0':
                if cur_stat and not prev_stat:
                    models.test_connect.acquisition_start()
                    RGB = models.streams[0].get_buffer().get_image().get_data()
                    RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)
                    models.test_connect.acquisition_stop()

                    inference(RGB)
                    
                    del RGB

                prev_stat = cur_stat
                cur_stat = True

            elif ser_value == b'1':
                prev_stat = cur_stat
                cur_stat = False
                
        except Exception as e:
            message = traceback.format_exc()
            logger.info(message)
            send_message_to_slack("----- [Unit 1] cam1 error -----\n" + message)
            send_message_to_slack(str(models.test_connect.get_feature('DeviceTemperature').value))
            models.test_connect.acquisition_stop()
            models.test_connect.dis_connect()
            del RGB
            break
