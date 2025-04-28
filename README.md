This is the Pytorch implementation for MambaGM

### Environment

pip install -r requirements.txt

### Data

Download from Google Drive: [Baby/Sports/Clothing/Electronics](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG).
The data contains text and image features extracted from Sentence-Transformers and VGG-16 and has been publiced in [MMRec](https://github.com/enoche/MMRec) framework.

### Run

1. Put your downloaded data (e.g. baby) under `data/` dir.
2. Run `main.py` to train MambaGM:
  `python main.py`
You may specify other parameters in CMD or config with `configs/model/*.yaml` and `configs/dataset/*.yaml`.

#### Acknowledgements

Thanks Zhou _et.al_ [FREEDOM](https://github.com/enoche/FREEDOM)),for their open source code.
This code is also based on [MMRec](https://github.com/enoche/MMRec). Thanks for their work.
