# Machine-Learning-Foundations
UTS Micro Credential Course - Machine Learning Foundations Assignment repository

The code implements a basic implementation of the adaboost algorithm and can achieve approximately 86%-88% accuracy but has a high false classification rate for mis-classifying “acceptable” wine as “unacceptable” wine.

The model did under-perform in false classifications of “acceptable” wines in the testing case where “acceptable” was a quality of 6.5 or higher.  The variance of this is, as shown in Table 4. Multiple Run Results, is from 19 to 38 of the 42 correct samples.

The model was significantly poor in providing statistics to understand how the model was performing and the ability to define the features of importance.  It is seen in the sklearn AdaBoost Classifier that there the capability to retrieve and plot this detail.
