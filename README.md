Work in progress.

Run the code!
--------
To promote reproducability we have provided a Dockerfile build on the tensorflow/tensorflow:2.1.0-gpu-py3 image.
For information on how to install docker please refer to https://docs.docker.com/get-docker/


Once you have docker installed run the following command to build the docker image:

```
docker build -f Dockerfile . -t pdtyping
```


**OPTIONAL** To run a terminal in the container:

```
docker run --rm -it -v $(pwd):/opt/project pdtyping bash
```

Alternatively you can run scripts directly with the docker container via the run command.
Note that the following examples assumes that you have navigated to the repository root directory.

To download the *"Online English"* dataset into to the correct location run:

```
docker run --rm -it -v $(pwd):/opt/project pdtyping bash ./download_data.sh
```


After downloading the raw data, execute the data preprocessing script 

```
docker run --rm -it -v $(pwd):/opt/project pdtyping python preprocess.py
```


To reproduce the results from the "Online English" dataset presented in Table 2 execute the run_experiments.py
command line script with the corresponding flag.

`
-e timeandchar
` *Time and Character (one-hot)* 

`
-e timeonly
` *Time Only*

`
-e word2vec 
` *Time and Character (CBOW)*

E.g. to reproduce the "Time and Character (one-hot)" experiment for the "Online English" dataset run

```
docker run --rm -it -v $(pwd):/opt/project pdtyping python run_experiments.py -e timeandchar
```

The resulting predictions and logs will be written to results/conll2020_MRC_{experiment flags}.
To analyse and plot the results execute `evaluate.py` with flag `-d` pointing to the results directory. To split results for medicated and unmedicated subjects as in Table 2 use flag `-ms`


```
docker run --rm -it -v $(pwd):/opt/project pdtyping python evaluate.py -d results/conll2020_MRC_P-robust_G-robust_S-0_standardconll2020 -ms
```


Finally if you wish to produce GradCam visualisations for the experiment
run `produce_gradcams.py` with flag `-d` pointing to the results directory. 


```
docker run --rm -it -v $(pwd):/opt/project pdtyping python produce_gradcams.py -d results/conll2020_MRC_P-robust_G-robust_S-0_standard
```

