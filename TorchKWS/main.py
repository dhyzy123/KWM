"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/3/15 下午6:07
@Author  : Yang "Jan" Xiao 
@Description : main
"""
import os
import logging.config
import time
import argparse
from utils.utils import *
from utils.train_utils import *
from train import Trainer, get_dataloader_keyword

if __name__ == "__main__":
    def options():


        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=200, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
        parser.add_argument("--batch", default=128, type=int, help="Training batch size")
        parser.add_argument("--step", default=30, type=int, help="Training step size")
        parser.add_argument("--gpu", default=3, type=int, help="Number of GPU device")
        parser.add_argument("--root", default="/root/autodl-tmp/SpeechCommands", type=str, help="The path of dataset")
        parser.add_argument("--dataset", default="gsc_v1", help="The name of the data set")
        parser.add_argument("--model", default="kwm-192-12_v1", type=str, help="models")
        parser.add_argument("--save", default="weight_kwm-192-12_v1", type=str, help="The save name")
        parser.add_argument("--opt", default="adamw", type=str, help="The optimizer")
        parser.add_argument("--sche", default="cos", type=str, help="The scheduler")
        parser.add_argument("--wd", default=0.1, type=int, help="The weight_decay")
        parser.add_argument("--warmup", default=10, type=int, help="The warmup epoch")
        args = parser.parse_args()
        return args


    parameters = options()
    # torch.set_num_threads(1)
    """
    Data 
    """
    if parameters.dataset == "gsc_v1" or parameters.dataset == "gsc_v2":
        class_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
        class_encoding = {category: index for index, category in enumerate(class_list)}
    elif parameters.dataset == "gsc_v1_30":
        class_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", 
                      "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", 
                      "house", "marvin", "sheila", "tree", "wow"]
        class_encoding = {category: index for index, category in enumerate(class_list)}
    elif parameters.dataset == "gsc_v2_35":
        class_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", 
                      "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", 
                      "house", "marvin", "sheila", "tree", "wow", "backward", "forward", "follow", "learn", "visual"]
        class_encoding = {category: index for index, category in enumerate(class_list)}
    """
    Logger 
    """
    save_path = f"{parameters.dataset}/{parameters.model}_lr{parameters.lr}_epoch{parameters.epoch}_wd{parameters.wd}_batch{parameters.batch}"
    logging.config.fileConfig("/home/dhy/program/TorchKWS/logging.conf")
    logger = logging.getLogger()
    os.makedirs(f"/root/autodl-tmp/Result/logs/{parameters.dataset}", exist_ok=True)
    fileHandler = logging.FileHandler("/root/autodl-tmp/Result/logs/{}.log".format(save_path), mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(f"[1] Select a KWS dataset ({parameters.dataset})")
    """
    Dataloader
    """
    data_path = os.path.join(parameters.root, parameters.dataset)
    logger.info(f"[2] Load the KWS dataset from {data_path}")
    train_loader, valid_loader, test_loader = get_dataloader_keyword(data_path, class_list,
                                                       class_encoding, parameters, noise_aug=True)    
    """
    Model 
    """
    model = select_model(parameters.model, len(class_list))
    logger.info(f"[3] Select a KWS model ({parameters.model})")
    logger.info(f"{parameters.model} parameters: {parameter_number(model)}")
    total_iters_s = len(train_loader) * max(1, (parameters.epoch - parameters.warmup))
    total_iters_w = len(train_loader) * parameters.warmup
    optimizer, scheduler = select_optimizer(parameters.opt, parameters.lr, model, parameters.sche,
                                            parameters.wd, total_iters_s, total_iters_w, parameters.warmup)
    """
    Train 
    """
    start_time = time.time()
    trainer = Trainer(parameters, model, len(class_list))
    trainer.model_train(optimizer=optimizer, scheduler=scheduler,
                                                    train_dataloader=train_loader,
                                                    valid_dataloader=valid_loader)
    result = trainer.model_test(test_dataloader=test_loader)
    """
    Summary
    """
    # Total time (T)
    duration = time.time() - start_time
    logger.info(f"======== Summary =======")
    logger.info(f"{parameters.model} parameters: {parameter_number(model)}")
    logger.info(f"Total time {duration}, Avg: {duration / parameters.epoch}s")

