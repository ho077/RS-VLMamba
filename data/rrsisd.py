import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import transforms as T
from bert.tokenization_bert import BertTokenizer

import h5py
from refer.refer import REFER

from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()
def get_transform(args):
    transforms = [
                  T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]
    return T.Compose(transforms)

def add_random_boxes(img, min_num=20, max_num=60, size=32):
    h,w = size, size
    img = np.asarray(img).copy()
    img_size = img.shape[1]
    boxes = []
    num = random.randint(min_num, max_num)
    for k in range(num):
        y, x = random.randint(0, img_size-w), random.randint(0, img_size-h)
        img[y:y+h, x: x+w] = 0
        boxes. append((x,y,h,w) )
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        # print(args.refer_data_root)
        # exit(0)
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        self.max_tokens = 20
        self.target_cls = {"airplane", "airport", "golf field", "expressway service area", "baseball field","stadium",
                      "ground track field", "storage tank", "basketball court", "chimney", "tennis court", "overpass",
                      "train station", "ship", "expressway toll station", "dam", "harbor", "bridge", "vehicle",
                      "windmill"}
        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        num_images_to_mask = int(len(ref_ids) * 0.2)
        self.images_to_mask = random.sample(ref_ids, num_images_to_mask)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name']))
        if self.split == 'train' and this_ref_id in self.images_to_mask:
            img = add_random_boxes(img)

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img, annot)

        sentence = ref[0]['sentences'][0]['raw']
        save_prefix = str(ref[0]['image_id']) + "_" + sentence

        if self.eval_mode:
            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]
        refer_text = ref[0]['sentences'][0]['raw']

        return img, target, tensor_embeddings, attention_mask,save_prefix,refer_text
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    # 模拟命令行参数
    parser = get_parser()
    parser.set_defaults(
        refer_data_root="/hy-tmp/RRSIS-D",  # 请替换为你的实际路径
        img_size=512,
        bert_tokenizer="bert-base-uncased"
    )
    args = parser.parse_args()

    # 构建 transform
    transform = get_transform(args)

    # 创建数据集
    dataset = ReferDataset(
        args=args,
        image_transforms=transform,
        split='train',      # 可改为 'train' 测试训练集
        eval_mode=False   # 若测试训练过程行为，设为 False
    )

    print(f"Dataset size: {len(dataset)}")

    # 创建 DataLoader（batch_size=4）
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,     # 调试时建议关闭 shuffle，便于复现
        num_workers=0,     # 调试时建议设为 0，避免多进程掩盖错误
        pin_memory=False,
        drop_last=False
    )

    print("\n🔍 Testing DataLoader with batch_size=4...\n")

    for batch_idx, batch in enumerate(dataloader):
        img_batch, target_batch, input_ids_batch, attention_mask_batch = batch
        input_ids_batch = input_ids_batch.squeeze(1)
        attention_mask_batch = attention_mask_batch.squeeze(1)

        print(f"Batch {batch_idx}:")
        print(f"  Images:      {img_batch.shape} (dtype={img_batch.dtype})")
        print(f"  Targets:     {target_batch.shape} (dtype={target_batch.dtype})")
        print(f"  Input IDs:   {input_ids_batch.shape} (dtype={input_ids_batch.dtype})")
        print(f"  Attn Mask:   {attention_mask_batch.shape} (dtype={attention_mask_batch.dtype})")

        # 检查是否有 NaN 或异常值
        if torch.isnan(img_batch).any():
            print("  ❌ Warning: NaN in images!")
        if torch.isnan(target_batch).any():
            print("  ❌ Warning: NaN in targets!")
        if (target_batch < 0).any() or (target_batch > 1).any():
            print(f"  ⚠️  Target values out of [0,1]: min={target_batch.min().item()}, max={target_batch.max().item()}")

        # 只测试前 3 个 batch（避免刷屏）
        if batch_idx >= 2:
            break

    print("\n✅ DataLoader test completed.")