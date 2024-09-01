import os.path

import cv2
import torch
from torchvision import transforms
from arch_unet import UNet
from train import validation_kodak, AugmentNoise, calculate_psnr, calculate_ssim
import numpy as np
from tqdm import trange

def denoise_result(data_dir,model,output_dir,noise_type):
    # 数据集载入
    valid_dataset = validation_kodak(data_dir)
    noise_adder = AugmentNoise(style=noise_type) # "gauss25"
    psnr_result = []
    ssim_result = []

    for idx, img in enumerate(valid_dataset):
        # 添加噪声
        origin255 = img.copy()
        origin255 = origin255.astype(np.uint8)
        im = np.array(img, dtype=np.float32) / 255.0
        noisy_im = noise_adder.add_valid_noise(im)
        noisy255 = noisy_im.copy()
        noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                           255).astype(np.uint8)

        # padding to square
        H = noisy_im.shape[0]
        W = noisy_im.shape[1]
        val_size = (max(H, W) + 31) // 32 * 32
        noisy_im = np.pad(
            noisy_im,
            [[0, val_size - H], [0, val_size - W], [0, 0]],
            'reflect')
        transformer = transforms.Compose([transforms.ToTensor()])
        noisy_im = transformer(noisy_im)
        noisy_im = torch.unsqueeze(noisy_im, 0)
        noisy_im = noisy_im.cuda()
        # 进行预测
        with torch.no_grad():
            prediction = model(noisy_im)
            prediction = prediction[:, :, :H, :W]

        prediction = prediction.permute(0, 2, 3, 1)
        prediction = prediction.cpu().data.clamp(0, 1).numpy()
        prediction = prediction.squeeze()
        pred255 = np.clip(prediction * 255.0 + 0.5, 0,
                          255).astype(np.uint8)
        # calculate psnr
        cur_psnr = calculate_psnr(origin255.astype(np.float32),
                                  pred255.astype(np.float32))
        psnr_result.append(cur_psnr)
        cur_ssim = calculate_ssim(origin255.astype(np.float32),
                                  pred255.astype(np.float32))
        ssim_result.append(cur_ssim)

        # 设置文本参数
        text = f"psnr:{round(cur_psnr, 2)} ssim:{round(cur_ssim, 2)}"  # 要添加的文本
        position = (50, 50)  # 文本在图像中的位置 (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
        font_scale = 1  # 字体大小
        color = (0, 0, 0)  # 文本颜色 (B, G, R)
        thickness = 2  # 文本线条的厚度

        res = np.hstack((origin255, noisy255, pred255))
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        # 在图像上添加文本
        cv2.putText(res, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imwrite(output_dir + f"/{idx}.png", res)

    with open(os.path.join(output_dir,"log.txt"),"w",encoding="utf-8") as fp:
        fp.writelines(f"noise_type:{noise_type} avg_psnr:{sum(psnr_result)/len(psnr_result)} avg_ssim:{sum(ssim_result)/len(ssim_result)}\n")

if __name__ == "__main__":
    device = torch.device("cuda:0")

    # 载入模型
    model = UNet().to(device)
    params = torch.load("./pretrained_model/model_gauss25_b4e100r02.pth")
    model.load_state_dict(params)
    dataset1 = r"D:\jupyter_notebook\paper\dataset\Kodak24"
    name1 = "Kodak24"
    dataset2 = r"D:\jupyter_notebook\paper\dataset\McMaster"
    noise_type_list = ["gauss10","gauss25","gauss50","poisson10","poisson25","poisson50"]
    name2 = "McMaster"
    for noise_type_idx in trange(len(noise_type_list)):
        res_path1 = name1 +'_'+noise_type_list[noise_type_idx]
        if not os.path.exists(res_path1):
            os.mkdir(res_path1)
        denoise_result(dataset1,model,name1 +'_'+noise_type_list[noise_type_idx],noise_type_list[noise_type_idx])

        res_path2 = name2 + '_' + noise_type_list[noise_type_idx]
        if not os.path.exists(res_path2):
            os.mkdir(res_path2)
        denoise_result(dataset2, model, name2+ '_' + noise_type_list[noise_type_idx], noise_type_list[noise_type_idx])


    pass
