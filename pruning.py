# This is only for DexiNed to prune and fine-tune
# For now only doing the pytorch unstructured prune

from model import DexiNed
import torch
import torch.nn.utils.prune as prune
import torch.optim as optim
from losses import *    # for importing bdcn_loss2
from datasets import DATASET_NAMES, BipedDataset, TestDataset, dataset_info     # for data loader
from torch.utils.data import DataLoader
import multiprocessing   # for getting the number of workers(cpu cores)
import argparse
import os
import kornia as kn
import cv2
import copy


"""
pytorch note:

model.named_modules(): 裡面儲存所有layer的module
module.named_buffers(): 裡面儲存該module的weight跟bias mask tensors
module.named_parameters(): 裡面儲存了該module的weight跟bias，如果有prune的話，
                           "weight"會變成"weight_orig"，"bias"會變成"bias_orig"。
                           如果沒有prune，名稱還是"weight"跟"bias"。
                           NOTE: 不管有沒有prune，它的weight跟bias都是原始的值(不含0)。
module.weight: 裡面儲存了pruned之後的weights，代表它包含了0(被pruned的weight會設成0)
"""

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
    """
    Calculate single module sparsity
    """
    num_zeros = 0
    num_elements = 0

    if use_mask:
        # module.named_buffers()裡面儲存了mask tensors(0跟1組成的matrix, 0表示要被prune掉的weight or bias)
        # tensor mask基本上又分成"weight_mask"跟"bias_mask"
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight:
                num_zeros += torch.sum(buffer==0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias:
                num_zeros += torch.sum(buffer==0).item()
                num_elements += buffer.nelement()
    else:
        # 如果有prune的話，module.named_parameters()裡面會儲存"weight_orig"或"bias_orig"
        # 如果沒有prune的話，module.named_parameters()裡面會儲存"weight"和"bias"
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight:
                num_zeros += torch.sum(param==0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias:
                num_zeros += torch.sum(param==0).item()
                num_elements += param.nelement()
    sparsity = num_zeros / num_elements
    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model, weight=True, bias=False, conv2d_use_mask=False, linear_use_mask=False):
    """
    Calculate global sparsity
    """
    num_zeros = 0
    num_elements = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements
    return num_zeros, num_elements, sparsity


def save_image_batch_to_disk(tensor, output_dir, file_names, img_shape=None):
    os.makedirs(output_dir, exist_ok=True)
    assert len(tensor.shape) == 4, tensor.shape
    img_shape = np.array(img_shape)
    # img_shape = img_shape.numpy()
    for tensor_image, file_name in zip(tensor, file_names):
        image_vis = kn.utils.tensor_to_image(torch.sigmoid(tensor_image))
        image_vis = (255.0 * (1.0 - image_vis))
        output_file_name = os.path.join(output_dir, file_name)
        image_vis = cv2.resize(image_vis, dsize=(img_shape[1], img_shape[0]))
        assert cv2.imwrite(output_file_name, image_vis)


def evaluate_model(model, test_loader, device, epoch=0, iteration=0):
    model.eval()
    output_dir = "outputs/CLASSIC"
    output_dir = os.path.join(output_dir, "iteration"+str(iteration)+"_epoch"+str(epoch))
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for _, sample_batched in enumerate(test_loader):
            images = sample_batched['images'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            preds = model(images)
            save_image_batch_to_disk(preds[-1], output_dir, file_names, img_shape=image_shape)
        print("finished saving images")


def train_model(model, train_loader, test_loader, device, l1_regularization_strength=0,
                l2_regularization_strength=1e-4, learning_rate=1e-4, num_epochs=2,
                batch_size=4, iteration=0):
    """
    Training the pruned model
    """
    criterion = bdcn_loss2
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    # evaluation, since the loss does not really matter for DexiNed
    # just save the result image to verify.
    evaluate_model(model=model, test_loader=test_loader, device=device)


    # fine-tuning
    print("Strat fine-tuning...")
    for epoch in range(num_epochs):
        print("epoch{}".format(epoch))
        model.train()

        avg_loss = []
        for batch_id, sample_batched in enumerate(train_loader):
            inputs = sample_batched['images'].to(device)
            labels = sample_batched['labels'].to(device)
            preds_list = model(inputs)

            # zero out the parameter gradients
            optimizer.zero_grad()
            l_weight = [0.7,0.7,1.1,1.1,0.3,0.3,1.3]
            loss = sum([criterion(preds, labels, l_w)/batch_size for preds, l_w in zip(preds_list,l_weight)])

            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
        avg_loss = np.array(avg_loss).mean()
        print("Epoch{:03d}: Train Loss: {:.5f}".format(epoch+1, avg_loss))

        # Evaluation for every epoch
        evaluate_model(model=model, test_loader=test_loader, device=device, epoch=epoch+1, iteration=iteration)

    print("Finished fine-tuning")
    return model


def iterative_pruning_finetuning(model, train_loader, test_loader, iterations=10, batch_size=4):
    for i in range(iterations):
        """
        Pruning
        Prune the weights in conv layers
        """
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # print("module name: ", module_name)
                # print(module)
                parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method = prune.L1Unstructured,
            amount = 0.2    # amount of rate(0-1) to be pruned
        )

        num_zeros, num_elements, sparsity = measure_global_sparsity(model, weight=True, bias=False, conv2d_use_mask=True, linear_use_mask=False)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))


        """
        Fine-tuning
        """
        learning_rate = 1e-4
        num_epochs = 2
        model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=device,
                            l1_regularization_strength=0, l2_regularization_strength=1e-4, learning_rate=learning_rate,
                            num_epochs=num_epochs, batch_size=batch_size, iteration=i)
    return model


def prepare_dataloader(batch_size):
    """
    prepare for the training and testing dataloader
    """
    num_workers = multiprocessing.cpu_count()
    input_dir = "data/magnet_ref"

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, choices=DATASET_NAMES, default="CLASSIC")
    parser.add_argument('--train_list', type=str, default='train_pair.lst')
    args = parser.parse_args()

    train_dataset = BipedDataset(input_dir, img_width=368, img_height=240, mean_bgr=[103.939,116.779,123.68], train_mode='train', arg=args)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = TestDataset(input_dir, test_data="CLASSIC", img_width=368, img_height=240, mean_bgr=[103.939,116.779,123.68], test_list=None, arg=args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model



if __name__ == "__main__":
    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

    # Instantiate model and move it to the computing device
    model = DexiNed()

    # Load the model
    checkpoint_path = "439_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    batch_size = 4
    train_loader, test_loader = prepare_dataloader(batch_size)

    # module_1 = model.block_1.conv1
    # print(list(module_1.named_parameters()))

    # iterative pruning and finetuning
    pruned_model = copy.deepcopy(model)
    pruned_model.to(device)
    pruned_model = iterative_pruning_finetuning(pruned_model, train_loader, test_loader, iterations=10, batch_size=4)

    # Apply mask to the parameters and remove the mask
    model = remove_parameters(pruned_model)

    evaluate_model(model=model, test_loader=test_loader, device=device, epoch=1000, iteration=1000)

    num_zeros, num_elements, sparsity = measure_global_sparsity(model)
    print("Global Sparsity:")
    print("{:.2f}".format(sparsity))

    model_dir = "saved_model"
    model_filename = "{}_iterations10_sparsity_{:.2f}.pt".format("pruned_model", sparsity)
    model_filepath = os.path.join(model_dir, model_filename)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_filepath)

    print("ALL DONE")




"""
# check the sparsity in each conv layer in DexiNed
print("Sparsity in block_1.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.block_1.conv1.weight == 0)) / float(model.block_1.conv1.weight.nelement())))
print("Sparsity in block_1.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.block_1.conv2.weight == 0)) / float(model.block_1.conv2.weight.nelement())))
print("Sparsity in block_2.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.block_2.conv1.weight == 0)) / float(model.block_2.conv1.weight.nelement())))
print("Sparsity in block_2.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.block_2.conv2.weight == 0)) / float(model.block_2.conv2.weight.nelement())))
print("Sparsity in dblock_3.denselayer1.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_3.denselayer1.conv1.weight == 0)) / float(model.dblock_3.denselayer1.conv1.weight.nelement())))
print("Sparsity in dblock_3.denselayer1.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_3.denselayer1.conv2.weight == 0)) / float(model.dblock_3.denselayer1.conv2.weight.nelement())))
print("Sparsity in dblock_3.denselayer2.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_3.denselayer2.conv1.weight == 0)) / float(model.dblock_3.denselayer2.conv1.weight.nelement())))
print("Sparsity in dblock_3.denselayer2.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_3.denselayer2.conv2.weight == 0)) / float(model.dblock_3.denselayer2.conv2.weight.nelement())))
print("Sparsity in dblock_4.denselayer1.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_4.denselayer1.conv1.weight == 0)) / float(model.dblock_4.denselayer1.conv1.weight.nelement())))
print("Sparsity in dblock_4.denselayer1.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_4.denselayer1.conv2.weight == 0)) / float(model.dblock_4.denselayer1.conv2.weight.nelement())))
print("Sparsity in dblock_4.denselayer2.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_4.denselayer2.conv1.weight == 0)) / float(model.dblock_4.denselayer2.conv1.weight.nelement())))
print("Sparsity in dblock_4.denselayer2.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_4.denselayer2.conv2.weight == 0)) / float(model.dblock_4.denselayer2.conv2.weight.nelement())))
print("Sparsity in dblock_4.denselayer3.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_4.denselayer3.conv1.weight == 0)) / float(model.dblock_4.denselayer3.conv1.weight.nelement())))
print("Sparsity in dblock_4.denselayer3.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_4.denselayer3.conv2.weight == 0)) / float(model.dblock_4.denselayer3.conv2.weight.nelement())))
print("Sparsity in dblock_5.denselayer1.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_5.denselayer1.conv1.weight == 0)) / float(model.dblock_5.denselayer1.conv1.weight.nelement())))
print("Sparsity in dblock_5.denselayer1.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_5.denselayer1.conv2.weight == 0)) / float(model.dblock_5.denselayer1.conv2.weight.nelement())))
print("Sparsity in dblock_5.denselayer2.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_5.denselayer2.conv1.weight == 0)) / float(model.dblock_5.denselayer2.conv1.weight.nelement())))
print("Sparsity in dblock_5.denselayer2.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_5.denselayer2.conv2.weight == 0)) / float(model.dblock_5.denselayer2.conv2.weight.nelement())))
print("Sparsity in dblock_5.denselayer3.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_5.denselayer3.conv1.weight == 0)) / float(model.dblock_5.denselayer3.conv1.weight.nelement())))
print("Sparsity in dblock_5.denselayer3.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_5.denselayer3.conv2.weight == 0)) / float(model.dblock_5.denselayer3.conv2.weight.nelement())))
print("Sparsity in dblock_6.denselayer1.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_6.denselayer1.conv1.weight == 0)) / float(model.dblock_6.denselayer1.conv1.weight.nelement())))
print("Sparsity in dblock_6.denselayer1.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_6.denselayer1.conv2.weight == 0)) / float(model.dblock_6.denselayer1.conv2.weight.nelement())))
print("Sparsity in dblock_6.denselayer2.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_6.denselayer2.conv1.weight == 0)) / float(model.dblock_6.denselayer2.conv1.weight.nelement())))
print("Sparsity in dblock_6.denselayer2.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_6.denselayer2.conv2.weight == 0)) / float(model.dblock_6.denselayer2.conv2.weight.nelement())))
print("Sparsity in dblock_6.denselayer3.conv1.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_6.denselayer3.conv1.weight == 0)) / float(model.dblock_6.denselayer3.conv1.weight.nelement())))
print("Sparsity in dblock_6.denselayer3.conv2.weight: {:.2f}%".format(100. * float(torch.sum(model.dblock_6.denselayer3.conv2.weight == 0)) / float(model.dblock_6.denselayer3.conv2.weight.nelement())))
print("Sparsity in side_1.conv.weight: {:.2f}%".format(100. * float(torch.sum(model.side_1.conv.weight == 0)) / float(model.side_1.conv.weight.nelement())))
print("Sparsity in side_2.conv.weight: {:.2f}%".format(100. * float(torch.sum(model.side_2.conv.weight == 0)) / float(model.side_2.conv.weight.nelement())))
print("Sparsity in side_3.conv.weight: {:.2f}%".format(100. * float(torch.sum(model.side_3.conv.weight == 0)) / float(model.side_3.conv.weight.nelement())))
print("Sparsity in side_4.conv.weight: {:.2f}%".format(100. * float(torch.sum(model.side_4.conv.weight == 0)) / float(model.side_4.conv.weight.nelement())))
print("Sparsity in side_5.conv.weight: {:.2f}%".format(100. * float(torch.sum(model.side_5.conv.weight == 0)) / float(model.side_5.conv.weight.nelement())))
print("Sparsity in pre_dense_3.conv.weight: {:.2f}%".format(100. * float(torch.sum(model.pre_dense_3.conv.weight == 0)) / float(model.pre_dense_3.conv.weight.nelement())))
print("Sparsity in pre_dense_4.conv.weight: {:.2f}%".format(100. * float(torch.sum(model.pre_dense_4.conv.weight == 0)) / float(model.pre_dense_4.conv.weight.nelement())))
print("Sparsity in pre_dense_5.conv.weight: {:.2f}%".format(100. * float(torch.sum(model.pre_dense_5.conv.weight == 0)) / float(model.pre_dense_5.conv.weight.nelement())))
print("Sparsity in pre_dense_6.conv.weight: {:.2f}%".format(100. * float(torch.sum(model.pre_dense_6.conv.weight == 0)) / float(model.pre_dense_6.conv.weight.nelement())))
print("Sparsity in up_block_1.features.0.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_1.features[0].weight == 0)) / float(model.up_block_1.features[0].weight.nelement())))
print("Sparsity in up_block_2.features.0.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_2.features[0].weight == 0)) / float(model.up_block_2.features[0].weight.nelement())))
print("Sparsity in up_block_3.features.0.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_3.features[0].weight == 0)) / float(model.up_block_3.features[0].weight.nelement())))
print("Sparsity in up_block_3.features.3.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_3.features[3].weight == 0)) / float(model.up_block_3.features[3].weight.nelement())))
print("Sparsity in up_block_4.features.0.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_4.features[0].weight == 0)) / float(model.up_block_4.features[0].weight.nelement())))
print("Sparsity in up_block_4.features.3.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_4.features[3].weight == 0)) / float(model.up_block_4.features[3].weight.nelement())))
print("Sparsity in up_block_4.features.6.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_4.features[6].weight == 0)) / float(model.up_block_4.features[6].weight.nelement())))
print("Sparsity in up_block_5.features.0.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_5.features[0].weight == 0)) / float(model.up_block_5.features[0].weight.nelement())))
print("Sparsity in up_block_5.features.3.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_5.features[3].weight == 0)) / float(model.up_block_5.features[3].weight.nelement())))
print("Sparsity in up_block_5.features.6.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_5.features[6].weight == 0)) / float(model.up_block_5.features[6].weight.nelement())))
print("Sparsity in up_block_5.features.9.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_5.features[9].weight == 0)) / float(model.up_block_5.features[9].weight.nelement())))
print("Sparsity in up_block_6.features.0.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_6.features[0].weight == 0)) / float(model.up_block_6.features[0].weight.nelement())))
print("Sparsity in up_block_6.features.3.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_6.features[3].weight == 0)) / float(model.up_block_6.features[3].weight.nelement())))
print("Sparsity in up_block_6.features.6.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_6.features[6].weight == 0)) / float(model.up_block_6.features[6].weight.nelement())))
print("Sparsity in up_block_6.features.9.weight: {:.2f}%".format(100. * float(torch.sum(model.up_block_6.features[9].weight == 0)) / float(model.up_block_6.features[9].weight.nelement())))
print("Sparsity in block_cat.conv.weight: {:.2f}%".format(100. * float(torch.sum(model.block_cat.conv.weight == 0)) / float(model.block_cat.conv.weight.nelement())))
"""
