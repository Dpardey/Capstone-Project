### Capstone Project

# Home Credit Default Risk

Credit stablisment exists to loan money to their customers. The financial system function as a lubricant for the society. Largely, they collect money and allocate it in, primarly, credits. 
Nonetheless allocate this resources wisely are true headache for the banking system, I mean, not a simple headache, is a migrane.
Sometimes this problem is occasioned by a gap in the financial information, either not enough credit history or lack of it. And, unfortunately, this population is often
taken advantage of by untrustworthy lenders that operates illegally and can comission whatever they want.

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. 
In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of 
alternative data--including telco and transactional information--to predict their clients' repayment abilities. 
Credit lending has to be well-balanced in orther to charge an appropiate fee. Risky loans has to have a higher fee relative to safer loans,
in part for provisions (money storage my the lender in a secure account that the regulators demand for each loan) and to absorb to possible 
scenario of a default (risk and return philosophy: Higher risk derive higher return).

This is where Artificial Intelligence, especially Machine Learning models, play a vital role. 
These algorithms have the capacity to learn from historical data, i.e. find patterns of the group of customer whom previosly defaulted and 
accurately return an score (probability). This credit default risk is what is then used to determine the interest rate in the loan.

### Files in the repository

**1. application_train.zip**:
     This is the dataset were all the analysis was established. It's in zip format because of the size. It's needed to unzip the dataset.

**2. application_test.zip**: 
     Dataset needed for submission in Kaggle. If the model wants to be scored with in kaggle, this table has all the same columns that application_train but without the target variable. This dataset is also en zipformat.
    
**3. HomeCredit_columns_description.csv**: 
     Dataset containing the details about every column across all datasets.
     
**4. HomeCredit_logo.jpg**:
     Home Credit logo
     
**5. MAE.gif**:
     Median Absolute Error formula
     
**6. ROC_curve.png**:
     ROC curve image
    
**7. Home Credit Default Risk.ipynb**:
     Jupyter notebook containing all the analysis.
     
**8. Home Credit Default Risk.py**:
     The same *Home Credit Default Risk.ipynb* file with .py format

