# Data Augmentation for Software Vulnerability Assessment

This is the README file for the reproduction package of the paper: "Mitigating Data Imbalance for Software Vulnerability Assessment: Does Data Augmentation Help?".

The package contains the following artefacts:
1. Data files:
	+ `outputs.csv`: containing the vulnerability descriptions and CVSS metrics for training the models with and without the data augmentation techniques, except Back Translation and Paraphrasing
	+ `outputs_backtranslate.csv`: containing the data used to train the models with the Back Translation data augmentation technique.
	+ `outputs_gpt.csv`: containing the data used to train the models with the Paraphrasing data augmentation technique.
2. Code: contains the source code we used in our work to answer Research Questions (RQs) 1 and 2. The instructions to run the code are given below.

Before running any code, please install all the required Python packages using the following command: `pip install -r requirements.txt`

To generate the results for all the configurations in the paper, please run the following code:

`python3 Code/main.py <f> <aug> <m> <out>`

+ f : 'TFIDF', 'Doc2Vec', 'DL' // Note that DL is used for CNN and LSTM
+ da: 'None', 'OS', 'US', 'DAI', 'DAD', 'DAS', 'DASyn', 'Comb', 'DABackTrans', 'DAParaphraseGPT'
+ m: 'RF', 'CNN', 'LSTM'
+ out: \['cvss2_access_vector','cvss2_access_complexity', 'cvss2_authentication', 'cvss2_integrity_impact', 'cvss2_availability_impact', 'cvss2_confidentiality_impact', 'cvss2_base_score'\]

Note that `cvss2_base_score` corresponds to the Severity of vulnerabilities. We would skip a sample if we could not back-translate or paraphrase it. Also, the files for back-translation and paraphrasing can be generated again to increase the diversity of the data used for training.

The results will be saved in the folder `results_cvss2/`.
