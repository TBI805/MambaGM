This is the Pytorch implementation for MambaFree

### Environment

pip install -r requirements.txt

### Data

Download from Google Drive: [Baby/Sports/Clothing](https://drive.google.com/drive/folders/1BxObpWApHbGx9jCQGc8z52cV3t9_NE0f?usp=sharing).
The data contains text and image features extracted from Sentence-Transformers and VGG-16 and has been publiced in [MMRec](https://github.com/enoche/MMRec) framework.

### Run

1. Put your downloaded data (e.g. baby) under `data/` dir.
2. Run `train.sh` to train MambaFree:
  `bash train.sh`
You may specify other parameters in CMD or config with `configs/model/*.yaml` and `configs/dataset/*.yaml`.

#### Acknowledgements

Thanks Guo _et.al_ [LGMRec](https://github.com/georgeguo-cn/LGMRec),for their open source code.
This code is also based on [MMRec](https://github.com/enoche/MMRec). Thank for their work.
