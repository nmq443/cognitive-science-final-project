# cognitive-science-final-project

Final project for Cognitive Science class (RBE3046 1) University of Engineering and Technology - Vietnam National University (UET-VNU).  

## Goal
Construct artificial EEG data using Generative adverserial networks (GAN) and then combine this data with original dataset to see if GAN can improve performance.

## How to run preprocess_data.ipynb:
Install BCI competition IV dataset 2a and save it in this directory with name `BCI2a-mat` with this format:

```
BCI2a-mat
│   ├── s1
│   │   ├── A01E.mat
│   │   └── A01T.mat
│   ├── s2
│   │   ├── A02E.mat
│   │   └── A02T.mat
│   ├── s3
│   │   ├── A03E.mat
│   │   └── A03T.mat
│   ├── s4
│   │   ├── A04E.mat
│   │   └── A04T.mat
│   ├── s5
│   │   ├── A05E.mat
│   │   └── A05T.mat
│   ├── s6
│   │   ├── A06E.mat
│   │   └── A06T.mat
│   ├── s7
│   │   ├── A07E.mat
│   │   └── A07T.mat
│   ├── s8
│   │   ├── A08E.mat
│   │   └── A08T.mat
│   └── s9
│       ├── A09E.mat
│       └── A09T.mat
```
