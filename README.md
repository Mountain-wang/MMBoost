# MMBoost
The source code for "Majority-to-Minority Resampling for Boosting-based Classification under Imbalanced Data"

# Abstract
Classification is a classical research field due to its broad applications in data mining such as event extraction, spam detection, and medical treatment. However, class imbalance is an unavoidable problem in many real-world applications. It is challenging for conventional learning algorithms to deal with imbalanced datasets, since they tend to be biased towards the majority class, while the minority class is crucial as well. Many previous studies have been explored to solve class imbalance, such as data sampling and class switching. In this paper, we propose a hybrid strategy named Majority-to-Minority Resampling (MMR) to select switched instances, which adaptively samples potential instances from the majority class to augment the minority class. To reduce the loss of information after sampling, we also propose a Majority-to-Minority Boosting (MMBoost) algorithm for classification by dynamically adjusting weights of the sampled instances. We conduct extensive experiments using real-world datasets. Experimental results demonstrate that the proposed framework achieves competitive performance for dealing with imbalanced data compared to several strong baselines across different common metrics.

# Architecture
this is the flow of our method
# Requirements
you can run it in any Python environment easily.
## Main packages
 Python           3.8.1  
 scikit-learn     0.24.2  
 imbalanced-learn 0.8.1  
 pandas           1.0.5  
 numpy            1.18.2  
## Steps to run our code:
1. Clone this repository: https://github.com/Mountain-wang/MMBoost.git  
2. Create a directory for this repository  
3. Save it in this directory  
4. You can run it on two ways:  
    - Run it in cmd, and follow the interactive guide . 'python mmbAPI.py'
    - Run it with your IDEA, and you can feel the source code clearly
    
# Dateset
The real world datasets we adopt in paper is from KEEL Repository and UCI Repository. You can download it by yourselves or seed email to us for acquiring the dataset quickly.

# Note
If you meet any problems, please email us or message me.
