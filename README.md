# Streaming_CCPP
Streaming Regressors for Electrical Power Prediction in a Combined Cycle Power Plant

-----------
DESCRIPTION
-----------
The prediction of electrical power is a key issue in combined cycle power plants (CCPPs). The power output of a plant can vary depending on environmental variables (temperature, pressure, humidity), and the challenge here arises whn we have to predict this power output as a function of these environmental conditions in order to maximize the profit from the available megawatt hours. With the application of machine learning techniques (i.e. regression algorithms), researchers have solved this problem and have managed to reduce the computational and time costs in comparison with the traditional thermodynamical analysis. This challenge has been tackled from a batch learning view hitherto, where data are assumed to be at rest, and where regression models do not continuously integrate new information into already constructed models. 

The approach here is closer to the Big Data and IoT paradigms in which data arrive continuously and where regression models have to learn incrementally. These scripts compare and examine the hourly electrical power prediction of some of the most used stream regression algorithms, and discuss about the best technique in terms of time processing and performance to be applied on this streaming scenario.

---------
THE CODE
---------
- Required frameworks: scikit-multiflow (https://scikit-multiflow.github.io/), scikit-garden (https://github.com/scikit-garden/scikit-garden/tree/master/skgarden), and scikit-learn
- Dependencies: pandas, numpy, math, warnings, pickle, scipy.io, scipy.stats, matplotlib.pyplot, seaborn, copy, timeit, datetime.

The "script.py" script is used to generate the final results. It uses the scikit-learn framework (PassiveAggressiveRegressor, SGDRegressor, and MLPRegressor techniques). It also uses the scikit-multiflow framework (RegressionHoeffdingTree and RegressionHAT techniques, and also the ADWIN drift detector). The framework scikit-garden is used for MondrianForestRegressor and MondrianTreeRegressor techniques.

The "evaluatePrequential_ENERGIA_v2.py" script is used for the streaming evaluation. This file should be placed in the corresponding folder of the scikit-multiflow package: '.../scikit-multiflow-master/src/skmultiflow/evaluation'

-----------
DATASET
-----------
The dataset is "CCPP_data.csv"
