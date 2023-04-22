python main.py --dataset cifar10 --model cct_2 --conv-size 3 --conv-layers 2 --checkpoint-path ./checkpoints/cct2-3x2_cifar10.pth data

python evaluate.py --dataset cifar10 --model cct_2 --conv-size 3 --conv-layers 2 --checkpoint-path ./checkpoints_downloaded/cct2-3x2_cifar10.pth data



python evaluate_text.py --dataset ag_news --model text_cct_2 --conv-size 1 --checkpoint-path ./checkpoints_downloaded/text_cct2-1_agnews_93.45.pth data_manual

# python evaluate_text.py --dataset trec --model text_cct_2 --conv-size 1 --checkpoint-path ./checkpoints_downloaded/text_cct2-1_trec_91.00.pth data


# python evaluate_text.py --dataset trec --model text_cct_2 --conv-size 2 --checkpoint-path  checkpoints_downloaded/text_cct2-2_trec_91.80.pth data

