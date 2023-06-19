python run.py --train-dir '/media/saket/Elements/datasets/feb2022demo/dataset/train' \
                --dev-dir '/media/saket/Elements/datasets/feb2022demo/dataset/dev' \
                --epochs 100 \
                --start-epoch 75 \
		        --lr 0.000005 \
                --batch-size 1 \
                --save-dir 'checlpoints/save' \
                --dataset-name 'Feb2022-demo' \
                --loss-name 'triplet' \
                --model-name 'vgg-embedding-siamese' \
                --optimizer-name 'Adam' \
		        --restore-from checlpoints/save/