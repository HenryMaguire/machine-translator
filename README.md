# Neural Machine Translation: An Introduction

## Introduction

A simple implementation of a Encoder-Decoder solution to French-English translation.

This initially started as my capstone project for the Udacity Machine Learning Nanodegree, where I built a sequence-to-sequence model in Tensorflow based on single-layer encoder-decoder architecture. I am continuing the study by trying to gauge how increasing the complexity of the model changes the accuracy of its predictions. Essentially, I aim to write a large review of as many different Neural Machine Translation architectures as I can.


## Instructions:

- Unzip the file "short_data.zip" and make sure the folder is called "short_data", with the pickle files in the directory below.
- If you like, navigate to the `text-preprocessing.ipynb` to investigate the preprocessing script. Be careful when running it as this will overwrite some of the `short_data` folder.
- Go to `machine-translator.ipynb`for the Neural Machine Translation code and the Benchmark model, plus some script to investigate the BLEU scores.
- A full write up to supplement the documentation in the ipython notebooks is available in `report.pdf`.
- `utils.py` contains helper functions, which occasionally give some insight into the workings of the NMT system so may be of interest to the user.

## Plans - adding complexity:

- More layers in the encoder and decoder - this is a failsafe method of improving the
                               accuracy of the model
- Attention mechanisms - I don't actually know much about these but hard Attention
                        has been shown to increase accuracy greatly, without adding
                        too many additional parameters.
- Bidirectional encoders and decoders - again, failsafe + not too many parameters.
- Investigate ConvNet and GAN implementations
