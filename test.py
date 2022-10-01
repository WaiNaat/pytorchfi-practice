import torch
import torchvision
import random
import copy
import numpy as np
import argparse

from torchvision import transforms
from tqdm import tqdm
from bitstring import BitArray

import vessl

import pytorchfi
#from pytorchfi import *
from pytorchfi.core import FaultInjection
from pytorchfi.neuron_error_models import random_neuron_location

from geturl import get_state_dict_url

vessl.init()

class custom_single_bit_flip(FaultInjection):
    def __init__(self, model, batch_size, flip_bit_pos=None, save_log_list=False, **kwargs):
        super().__init__(model, batch_size, **kwargs)
        self.flip_bit_pos = flip_bit_pos
        self.save_log_list = save_log_list

        self.log_original_value = []
        self.log_original_value_bin = []
        self.log_error_value = []
        self.log_error_value_bin = []
        self.log_bit_pos = []

    def reset_log(self):
        self.log_original_value = []
        self.log_original_value_bin = []
        self.log_error_value = []
        self.log_error_value_bin = []
        self.log_bit_pos = []

    def _single_bit_flip(self, orig_value, bit_pos):
        # data type 설정
        save_type = orig_value.dtype
        orig_value = orig_value.cpu().item()
        length = None
        if save_type == torch.float32:
            length = 32
        elif save_type == torch.float64:
            length = 64
        else:
            raise AssertionError(f'Unsupported Data Type: {save_type}')

        # single bit flip
        orig_arr = BitArray(float = orig_value, length = length)
        error = list(map(int, orig_arr.bin))
        error[bit_pos] = (error[bit_pos] + 1) % 2
        error = ''.join(map(str, error))
        error = BitArray(bin=error)
        new_value = error.float

        if self.save_log_list:
            self.log_original_value.append(orig_value)
            self.log_original_value_bin.append(orig_arr.bin)
            self.log_error_value.append(new_value)
            self.log_error_value_bin.append(error.bin)
            self.log_bit_pos.append(bit_pos)

        return torch.tensor(new_value, dtype=save_type)

    # structure from pytorchfi/neuron_error_models/single_bit_flip_func/single_bit_flip_signed_across_batch
    def neuron_single_bit_flip(self, module, input_val, output):
        corrupt_conv_set = self.corrupt_layer
        
        bits = output.dtype
        if bits == torch.float32:
            bits = 32
        elif bits == torch.float64:
            bits = 64
        else:
            raise AssertionError(f'Unsupported data type {bits}')
            
        if type(corrupt_conv_set) is list:
            inj_list = list(
                filter(
                    lambda x: corrupt_conv_set[x] == self.current_layer,
                    range(len(corrupt_conv_set)),
                )
            )
            for i in inj_list:
                self.assert_injection_bounds(index=i)
                prev_value = output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]]

                rand_bit = random.randint(0, bits - 1) if self.flip_bit_pos is None else self.flip_bit_pos

                new_value = self._single_bit_flip(prev_value, rand_bit)

                output[self.corrupt_batch[i]][self.corrupt_dim[0][i]][
                    self.corrupt_dim[1][i]
                ][self.corrupt_dim[2][i]] = new_value

        else:
            if self.current_layer == corrupt_conv_set:
                prev_value = output[self.corrupt_batch][self.corrupt_dim[0]][
                    self.corrupt_dim[1]
                ][self.corrupt_dim[2]]

                rand_bit = random.randint(0, bits - 1)

                new_value = self._single_bit_flip(prev_value, rand_bit)

                output[self.corrupt_batch][self.corrupt_dim[0]][self.corrupt_dim[1]][
                    self.corrupt_dim[2]
                ] = new_value     

        self.update_layer()
        if self.current_layer >= len(self.output_size):
            self.reset_current_layer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--input-path', type=str, default='/input',
                        help='input dataset path')
    parser.add_argument('--output-path', type=str, default='/output',
                        help='output files path')
    parser.add_argument('--detailed-log', action='store_true', default=False,
                        help='For saving detailed single bit flip log')
    args = parser.parse_args()

    # 실험 환경 설정
    experiment_id = 1
    model_name = "vgg11_bn"
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_" + model_name, pretrained=False)
    model.load_state_dict(torch.hub.load_state_dict_from_url(get_state_dict_url(model_name, 'cifar100')))
    save_dir = model_name + '_id' + str(experiment_id) + '_cifar100'

    seed = 12345678

    batch_size = 256
    img_size = 32
    channels = 3

    use_gpu = torch.cuda.is_available()

    corrupt_input_images = True
    save_detailed_results = True

    custom_bit_flip_pos = None
    layer_type = ['all']
    layer_nums = ['all']

    # https://hoya012.github.io/blog/reproducible_pytorch/
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # Transform statics from https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/vgg11_bn/default.log
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
        ]
    )

    data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # single bit flip을 일으킬 모델 만들기
    base_fi_model = custom_single_bit_flip(
        model = copy.deepcopy(model),
        batch_size = batch_size, 
        input_shape = [channels, img_size, img_size], 
        use_gpu = use_gpu,
        layer_types = layer_type,
        flip_bit_pos = custom_bit_flip_pos,
        save_log_list = save_detailed_results
    )

    print(base_fi_model.print_pytorchfi_layer_summary())

    # 실험 진행
    results = []
    error_logs = []

    for layer_num in range(1):
        
        orig_correct_cnt = 0
        orig_corrupt_diff_cnt = 0
        batch_idx = -1
        
        for images, labels in dataset:

            batch_idx += 1
            print(batch_idx)
            vessl.log({'batch_idx': batch_idx})
            '''
            if use_gpu:
                images = images.to(device='cuda')

            # 원본에 inference 진행
            model.eval()
            with torch.no_grad():
                orig_output = model(images)

            # single bit flip 위치 지정
            layer_num_list = []
            dim1 = []
            dim2 = []
            dim3 = []

            for _ in range(batch_size):
                layer, C, H, W = random_neuron_location(base_fi_model, layer=layer_num)

                layer_num_list.append(layer)
                dim1.append(C)
                dim2.append(H)
                dim3.append(W)

            # corrupted model 만들기
            base_fi_model.reset_log()
            corrupted_model = base_fi_model.declare_neuron_fault_injection(
                batch = [i for i in range(batch_size)],
                layer_num = layer_num_list,
                dim1 = dim1,
                dim2 = dim2,
                dim3 = dim3,
                function = base_fi_model.neuron_single_bit_flip
            )

            # corrupted model에 inference 진행
            corrupted_model.eval()
            with torch.no_grad():
                corrupted_output = corrupted_model(images)

            # 결과 정리
            original_output = torch.argmax(orig_output, dim=1).cpu().numpy()
            corrupted_output = torch.argmax(corrupted_output, dim=1).cpu().numpy()
            labels = labels.numpy()

            # 결과 비교: 원본이 정답을 맞춘 경우 중 망가진 모델이 틀린 경우를 셈
            for i in range(batch_size):

                if labels[i] == original_output[i]:
                    orig_correct_cnt += 1

                    if original_output[i] != corrupted_output[i]:
                        orig_corrupt_diff_cnt += 1

                        if save_detailed_results:
                            log = [
                                f'Layer: {layer_num}',
                                f'Batch: {batch_idx}',
                                f'Position: ({i}, {dim1[i]}, {dim2[i]}, {dim3[i]})',
                                f'Original value:  {base_fi_model.log_original_value[i]}',
                                f'Original binary: {base_fi_model.log_original_value_bin[i]}',
                                f'Flip bit: {base_fi_model.log_bit_pos[i]}',
                                f'Error value:     {base_fi_model.log_error_value[i]}',
                                f'Error binary:    {base_fi_model.log_error_value_bin[i]}',
                                f'Label:        {labels[i]}',
                                f'Model output: {corrupted_output[i]}',
                                '\n'
                            ]

                            error_logs.append('\n'.join(log))
            '''

        # 결과 저장
        if orig_correct_cnt == 0: orig_correct_cnt = 1
        result = f'Layer #{layer_num}: {orig_corrupt_diff_cnt} / {orig_correct_cnt} = {orig_corrupt_diff_cnt / orig_correct_cnt * 100:.4f}%, ' + str(base_fi_model.layers_type[layer_num]).split(".")[-1].split("'")[0]
        #print(result)
        results.append(result)

    # save log
    f = open('../output/' + save_dir + '.txt', 'w')

    f.write(base_fi_model.print_pytorchfi_layer_summary())
    f.write(f'\n\n===== Result =====\nSeed: {seed}\n')
    for result in results:
        f.write(result + '\n')

    f.close()