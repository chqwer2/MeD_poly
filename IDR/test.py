import argparse

import torch
import torch.nn.functional as F
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from utils import *
# from models import UNet_n2n_un
from models import build_model


def model_forward(net, noisy, padding=32):
    h, w, _ = noisy.shape
    pw, ph = (w + 31) // 32 * 32 - w, (h + 31) // 32 * 32 - h
    with torch.no_grad():
        input_var = torch.FloatTensor([noisy]).cuda().permute(0, 3, 1, 2)
        input_var = F.pad(input_var, (0, pw, 0, ph), mode='reflect')
        # print(input_var.shape,  noisy.shape, ph, pw)
        out_var = net(input_var)

    if pw != 0:
        out_var = out_var[..., :, :-pw]
    if ph != 0:
        out_var = out_var[..., :-ph, :]

    denoised = out_var.permute([0, 2, 3, 1])[0].detach().cpu().numpy()
    return denoised


rng = np.random.default_rng(seed=36)


def _bernoulli(p, shape):
    return torch.rand(shape) <= p


def gaussian(img, sigma):
    # + np.random.normal(0, self.sigma / 255.0, img_L2.shape)
    return img + torch.FloatTensor(img.shape).normal_(mean=0, std=sigma)


def gaussian_localvar(img):
    # var ** 0.5
    noise = np.random.normal(0, torch.sqrt(img * 255).numpy(), img.shape) / 255
    # print("noise max:", np.max(noise))
    # print("noise img:", torch.max(img)
    return img + noise  # torch.FloatTensor(img.shape).normal_(mean=0, std=torch.sqrt(img).numpy())


def salt_pepper(img, amount=0.05, salt_vs_pepper=0.5):
    flipped = _bernoulli(amount, img.shape)
    salted = _bernoulli(salt_vs_pepper, img.shape)
    peppered = ~salted

    img[flipped & salted] = 1
    img[flipped & peppered] = 0

    return img


def salt_pepper_3(img, amount=0.05, salt_vs_pepper=0.3):
    #
    flipped = _bernoulli(amount, img.shape[:2]).unsqueeze(-1).repeat(1, 1, 3)
    salted = _bernoulli(salt_vs_pepper, img.shape[:2]).unsqueeze(-1).repeat(1, 1, 3)
    # flipped = torch.repeat
    peppered = ~salted

    img[flipped & salted] = 1
    img[flipped & peppered] = 0

    return img, flipped & salted


def salt_pepper_3_torch(img, amount=0.05, salt_vs_pepper=0.3):
    peppered = _bernoulli(amount, img.shape[-2:]).unsqueeze(0).repeat(3, 1, 1)

    # salted = _bernoulli(salt_vs_pepper, img.shape[-2:]).unsqueeze(0).repeat(3, 1, 1)
    #
    # # flipped = torch.repeat
    # peppered = ~salted
    #
    # img[flipped & salted] = 1
    img[peppered] = 0

    return img  # , peppered


def poisson(img):
    img_ = (img.clone() * 255).to(torch.uint8)  # .uint8()
    # img = (scipy.misc.imread(filename)).astype(float)
    # noise_mask = numpy.random.poisson(img)

    vals = torch.unique(img_).shape[0]  # length
    vals = 2 ** torch.ceil(torch.as_tensor(np.log2(vals)))  # 255

    img = torch.poisson(img_ * vals) / float(vals) / 255

    return img


def speckle(img, sigma):
    return img + img * torch.FloatTensor(img.shape).normal_(mean=0, std=sigma)



def add_noise(clean, ntype, sigma=None):
    # assert ntype.lower() in ['gaussian', 'gaussian_gray', 'impulse', 'binomial', 'pattern1', 'pattern2', 'pattern3', 'line']
    assert sigma < 1
    img1 = torch.from_numpy(clean)
    img = img1.clone()
    print("img max:", torch.max(img))

    if ntype == "gaussian":
        print("gaussian")
        noisy = clean + np.random.normal(0, sigma, clean.shape)

    elif ntype == "poisson":
        noisy = poisson(img)

    elif ntype == "local_val":
        noisy = gaussian_localvar(img)

    elif ntype == "s&p":
        noisy = salt_pepper(img, amount=0.05, salt_vs_pepper=0.5)

    elif ntype == "speckle":
        noisy = speckle(img, sigma)

    elif ntype == 'binomial':
        h, w, c = clean.shape
        mask = np.random.binomial(n=1, p=(1 - sigma), size=(h, w, 1))
        noisy = clean * mask

    elif ntype == 'impulse':
        mask = np.random.binomial(n=1, p=(1 - sigma), size=clean.shape)
        noisy = clean * mask

    elif ntype[:4] == 'line':
        # sigma = 25 / 255.0
        h, w, c = clean.shape
        line_noise = np.ones_like(clean) * np.random.normal(0, sigma, (h, 1, 1))
        noisy = clean + line_noise

    elif ntype[:7] == 'pattern':
        # sigma = 5 / 255.0
        h, w, c = clean.shape
        n_type = int(ntype[7:])

        one_image_noise, _, _ = get_experiment_noise('g%d' % n_type, sigma, 0, (h, w, 3))
        noisy = clean + one_image_noise
    else:
        assert 'not support %s' % args.ntype

    return noisy




def test(args, net, test_data_path_set):
    for test_data_path in test_data_path_set:
        data_list = [os.path.join(test_data_path, item) for item in os.listdir(test_data_path) if
                     'jpg' in item or 'png' in item]

        # args.noise_types = [["s&p", 0], ["poisson", 0], ["local_val", 0],
        #                     ["speckle", 25], ["speckle", 50]]
        #
        args.noise_types = [['gaussian', 25], ['gaussian', 50], ['gaussian', 75], ['gaussian', 100]]

        for np1 in args.noise_types:
            noise_type = np1[0]
            noise_level = np1[1]

            # for noise_level in args.test_noise_levels:
            if args.save_img:
                save_dir = os.path.join(args.res_dir, '%s_n' % (args.ntype), 'sigma-%d' % (noise_level))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

            res = {'psnr': [], 'ssim': []}
            for idx, item in enumerate(data_list):
                gt = cv2.imread(item)
                if 'gray' in args.ntype:
                    gt = cv2.imread(item, 0)[..., np.newaxis]

                gt_ = gt.astype(float) / 255.0

                sigma = noise_level / 255. if noise_level > 1 else noise_level

                noisy = add_noise(gt_, args.ntype, sigma=sigma)

                if args.zero_mean:
                    noisy = noisy - 0.5

                print('\rprocess', idx, len(data_list), item.split('/')[-1], gt.shape, args.ntype, end='')
                denoised = model_forward(net, noisy)

                denoised = denoised + (0.5 if args.zero_mean else 0)
                denoised = np.clip(denoised * 255.0 + 0.5, 0, 255).astype(np.uint8)

                noisy = noisy + (0.5 if args.zero_mean else 0)
                noisy = np.clip(noisy * 255.0 + 0.5, 0, 255).astype(np.uint8)

                # save PSNR
                print("denoised, gt:", denoised.shape, gt.shape)
                temp_psnr = compare_psnr(denoised, gt, data_range=255)
                temp_ssim = compare_ssim(denoised, gt, data_range=255, channel_axis=-1)

                res['psnr'].append(temp_psnr)
                res['ssim'].append(temp_ssim)

                if args.save_img:
                    filename = item.split('/')[-1].split('.')[0] + '_%s' % args.ntype

                    cv2.imwrite(os.path.join(save_dir, '%s_%.2f_out.png' % (filename, temp_psnr)), denoised)
                    cv2.imwrite(os.path.join(save_dir, '%s_NOISY.png' % (filename)), noisy)
                    cv2.imwrite(os.path.join(save_dir, '%s_GT.png' % (filename)), gt)

            print('\r', 'noise type:', noise_type, 'noise lelvel', noise_level, test_data_path.split('/')[-1], len(data_list),
                  ', psnr  %.2f ssim %.3f' % (np.mean(res['psnr']), np.mean(res['ssim'])), args.ntype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='self supervised')
    parser.add_argument('--root', default="/mnt/lustre/share/cp/zhangyi3/", type=str)

    parser.add_argument('--ntype', default="gaussian", type=str, help='noise type')
    parser.add_argument('--model_path', default=None, type=str)

    parser.add_argument('--res_dir', default="results", type=str)
    parser.add_argument('--save_img', default=True, type=bool)
    args = parser.parse_args()

    args.zero_mean = False
    if args.model_path is None:
        args.model_path = 'checkpoint/%s.pth' % args.ntype

    print('Testing', args.model_path)

    # set testing noise levels
    if "gaussian" in args.ntype:
        args.zero_mean = True
        args.test_noise_levels = [25, 50]
    elif args.ntype == 'line':
        args.test_noise_levels = [25]
    elif args.ntype in ['binomial', 'impulse']:
        args.test_noise_levels = [0.5]
    else:
        args.test_noise_levels = [5]

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    # set testing datasets

    test_data_path_set = ["/bask/projects/j/jiaoj-3d-vision/Hao/data/denoising/CBSD68-dataset/CBSD68/original_png"]


    # model
    ch = 1 if 'gray' in args.ntype else 3
    # net = UNet_n2n_un(in_channels=ch, out_channels=ch)
    cfg = EasyDict()
    cfg.model_name = 'UNet_n2n_un'
    cfg.model_args = {'in_channels': ch, 'out_channels': ch}
    net = build_model(cfg)

    net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.load_state_dict(torch.load(args.model_path))
    net.eval()

    test(args, net, test_data_path_set)
