# libaom-AV1-POE
This repository contains a modified libaom encoder with a new perceptual encoding recipe.

## Introduction

This repository contains a modified version of the libaom encoder with a new perceptual encoding recipe. The new recipe is based on the following papers:

```bibtex
Pastor, Andréas, et al. "Predicting local distortions introduced by AV1 using Deep Features." 2023 IEEE International Conference on Visual Communications and Image Processing (VCIP). IEEE, 2023.
```

```bibtex
Pastor, Andréas, et al. "On the accuracy of open video quality metrics for local decision in av1 video codec." 2022 IEEE International Conference on Image Processing (ICIP). IEEE, 2022.
```

## Installation and Usage

To install the modified libaom encoder, follow the instructions below:

1. Clone the repository:
```bash
git clone
```

2. Build the modified libaom encoder:
```bash
cd aom
mkdir build-POE
cd build-POE
cmake .. -DCONFIG_A_MODE=1
make -j16
cd ../..
```
Where `-DCONFIG_A_MODE=1` enables the new perceptual encoding recipe, and `-j16` specifies the number of threads to use for the build.

3. Prepare the example sequence for encoding with our libaom-POE to YUV format:
```bash
wget https://media.xiph.org/video/aomctc/test_set/a2_2k/OldTownCross_1920x1080p50.y4m -O ./sequences/OldTownCross_1920x1080p50.y4m

ffmpeg -i ./sequences/OldTownCross_1920x1080p50.y4m ./sequences/OldTownCross_1920x1080p50.yuv
ffmpeg -i ./sequences/OldTownCross_1920x1080p50.y4m -c:v libx264 -crf 0 -pix_fmt yuv420p -an ./sequences/OldTownCross_1920x1080p50.mp4
```
converting the sequence "OldTownCross_1920x1080p50.y4m" to YUV format for encoding with our libaom-POE and put in a mp4 container for the lambda correction map prediction.

4. Generate the lambda correction map for the sequence:
```bash
python3 -m venv myenv
source myenv/bin/activate
python -m pip install opencv-python torch tqdm

mkdir encodes
mkdir external_data_for_encode

cd lambda_map_predictor

python predict_DL_raw.py --dist_path ../sequences/OldTownCross_1920x1080p50.mp4 --ref_path ../sequences/OldTownCross_1920x1080p50.mp4 --width 1920 --height 1080

cd ..
```



5. Run the modified libaom encoder using a pre-computed lambda correction map and a YUV sequence:

```bash
./aom/build-POE/aomenc -o ./encodes/OldTownCross_1920x1080p50_tune_llvq_57_0_inter.obu --verbose --psnr --lag-in-frames=16 --test-decode=fatal --obu --passes=1 --cpu-used=0 --i420 --width=1920 --height=1080 --fps=50/1 --input-bit-depth=8 --bit-depth=8 --end-usage=q --cq-level=57 --limit=65 --tile-columns=0 --threads=4 --use-fixed-qp-offsets=1 --deltaq-mode=0 --enable-tpl-model=0 --kf-min-dist=65 --enable-keyframe-filtering=0 --min-gf-interval=16 --max-gf-interval=16 --auto-alt-ref=1 --gf-min-pyr-height=4 --gf-max-pyr-height=4 --kf-max-dist=65 --tune=ssim --dump-folder=./external_data_for_encode/OldTownCross/dl2_rawlin21/ ./sequences/OldTownCross_1920x1080p50.yuv 2>&1 | tee ./logs/encoded_sequence-log.txt
```

## References

```bibtex
@inproceedings{pastor2023predicting,
  title={Predicting local distortions introduced by AV1 using Deep Features},
  author={Pastor, Andr{\'e}as and Krasula, Luk{\'a}{\v{s}} and Zhu, Xiaoqing and Li, Zhi and Le Callet, Patrick},
  booktitle={2023 IEEE International Conference on Visual Communications and Image Processing (VCIP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}


@inproceedings{pastor2022accuracy,
  title={On the accuracy of open video quality metrics for local decision in av1 video codec},
  author={Pastor, Andr{\'e}as and Krasula, Luk{\'a}{\v{s}} and Zhu, Xiaoqing and Li, Zhi and Le Callet, Patrick},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={4013--4017},
  year={2022},
  organization={IEEE}
}

```

## Presentations

The following presentations are available discussing the new perceptual encoding recipe and the perceptual gains obtained in a first subjective evaluation:

- [VQEG Summer 2024: Subjective Evaluation of Perceptually Optimized Video Encoding Recipes](https://docs.google.com/presentation/d/122_T0XsT-dpn9CvbN9OpKUbOQ0PZH_zsGwC-ppKDMZw/edit?usp=sharing)

- [Local Visual Perceptual Differences in Video: Psychophysical methods, computational prediction, and application to Perceptual Encoding in Open Video Codecs
](https://docs.google.com/presentation/d/1MaU2xOq6ZPDpVJjVIg5H4gnGALd6lVWxFRi-QaQdUxo/edit?usp=sharing)
