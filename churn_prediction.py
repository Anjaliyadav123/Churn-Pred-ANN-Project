import pandas as pd                    #Used for reading and manipulating data.
import numpy as np                      #Used for mathematical and numerical operations.
import matplotlib.pyplot as plt                   #Used to create charts and graphs.
import seaborn as sns                              #Used for advanced and attractive visualizations.
from sklearn.model_selection import train_test_split             #Splits data into training and testing sets.
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder    # MinMaxScaler- Scales values between 0 and 1. & OneHotEncoder- Converts categorical columns into numerical values.
from sklearn.metrics import classification_report                #Evaluates model performance.
from sklearn.utils.class_weight import compute_class_weight            #Class Weights- Handles imbalanced datasets.
import tensorflow as tf                        #Deep Learning library used to build Neural Networks.
from keras import Sequential, layers           #Sequential- Creates a neural network layer by layer. & Layers- Adds neural network layers.


df = pd.read_csv('D:/OneDrive - Encore Asset Reconstruction Company Pvt Ltd/Downloads/Churn_Modelling.csv')        #pd = pandas library, read_csv() = reads a CSV file, df = DataFrame (a table of rows and columns)
df.sample(5)          #sample() returns random rows from the dataset. & 5 means return 5 random records.


df.info()      #df.info() gives a summary of the DataFrame.

df.drop(columns= ['RowNumber','CustomerId', 'Surname'] ,axis= 1, inplace= True) 

#This line is used to remove unnecessary columns from your dataset.
#drop() is a Pandas function used to remove rows or columns.
#These are the columns you want to delete.
#axis=1 - axis=0 → rows & axis=1 → columns
#inplace=True - This updates the original DataFrame directly.Without inplace=True: The original df remains unchanged.



df.dtypes

#df.dtypes shows the data type of each column in your DataFrame.
#int64 - Whole numbers
#float64 - Decimal numbers
#object - Text/String values


df.columns

#df.columns displays the names of all columns in your DataFrame.
#list(df.columns)
#len(df.columns) - Meaning there are 14 columns.



for col in df.columns:
    print(f'{col} : {df[col].unique()}')

#This loop goes through every column in your DataFrame and prints all the unique values present in that column.
#df.columns- Returns all column names
#for col in df.columns - Loops through each column one by one.
#df[col] - Selects that column.
#unique() - Returns distinct values in the column.
#f'{col} : ...' - Formats the output nicely.


df.Gender.replace({'Female' : 1, 'Male' : 0} , inplace=True)

#This line converts the Gender column from text values (Female, Male) into numeric values (1, 0).


df.Gender.unique

#You need to call the function with parentheses.



# Splitting

X = df.drop('Exited', axis = 1)
y = df.Exited

#X = df.drop('Exited', axis=1) - Removes the Exited column from the dataset and stores all remaining columns in X.
#y = df.Exited - Selects only the target column.




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42, stratify=y)

#This line splits your dataset into training data and testing data.
#X - Your input features
#Y - Your target variable
#What train_test_split() does - Suppose your dataset has 10,000 rows. test_size=0.2, means:80% (8,000 rows) → Training data & 20% (2,000 rows) → Testing data
#X_train- Features used to train the model.
#X_test- Features used to test the model.
#y_train- Actual churn values for training.
#y_test - Actual churn values for testing.
#test_size=0.2 - Means: 20% of the data will be kept aside for testing. Ex - Total rows = 10000, Train = 8000 & Test  = 2000
#random_state=42 - Ensures you get the same split every time you run the code. Without it:the train and test records will change on every execution. The value 42 is just a commonly used number; any integer works.
#stratify=y - Maintains the same proportion of classes in both training and testing datasets.
         #Suppose:Exited = 0  → 8000 customers & Exited = 1  → 2000 customers, 
         #Percentage:Stayed = 80% & Exited = 20%
         #With: stratify=y, the split remains: For Training: 80% stayed & 20% exited & for Testing:80% stayed & 20% exited
#This step prepares your data so the model can learn from the training set and be evaluated on unseen test data.



y_test.value_counts()

#y_test.value_counts() counts how many times each class appears in your test dataset.




df.Exited.value_counts()    # We have Bias Here

#df.Exited.value_counts() counts how many customers belong to each class in the Exited column.





# and for Geography  => one hot encoding

encoder = OneHotEncoder(sparse_output= False, handle_unknown='ignore')
tr_geo = encoder.fit_transform(X_train[['Geography']])
te_geo = encoder.transform(X_test[['Geography']])

tr_df = pd.DataFrame(tr_geo, columns= encoder.get_feature_names_out(['Geography']), index= X_train.index)
te_df = pd.DataFrame(te_geo, columns= encoder.get_feature_names_out(['Geography']), index = X_test.index)

X_train = pd.concat([X_train.drop('Geography', axis= 1), tr_df], axis = 1)
X_test  = pd.concat([X_test.drop('Geography', axis = 1), te_df], axis = 1)


#This code is performing One-Hot Encoding on the Geography column and then adding the encoded columns back to your training and test datasets.
#Why is this needed? Suppose Geography contains:France, Germany, Spain, Machine learning models cannot directly understand text values, so we convert them into numeric columns.
#sparse_output=False - Returns a normal NumPy array.Without it:<Compressed Sparse Matrix> & With it:[[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
#handle_unknown='ignore' - If a new country appears in test data, the code won't crash. Training data:France, Germany, Spain & Test data: Italy. Without ignore:Error! & With ignore:[0 0 0]
#fit - Learns categories:France, Germany, Spain
#transform - Converts them into numeric columns. [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
#Transform Test Data - transform(), Not fit_transform() Because we only want to use the categories learned from the training set. This prevents data leakage.
#Convert Arrays to DataFrames for Training Data
#Creates the same columns for the test set.
#Remove Original Geography Column and Add Encoded Columns



X_train

#X_train is the training features dataset that will be used to train your machine learning model.
#What happened to Geography? - The original Geography column is removed and replaced with 3 new numeric columns.




# Scaling 
sc_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary','NumOfProducts']

scaler = MinMaxScaler()

X_train[sc_cols] = scaler.fit_transform(X_train[sc_cols])
X_test[sc_cols] = scaler.transform(X_test[sc_cols])



#This code is performing Feature Scaling using MinMaxScaler.
#Why Scale the Data? - Your columns have very different ranges:CreditScore:350-850, Age:18-92, Balance:0-250000, EstimatedSalary:11-200000, NumOfProducts:1-4
      #If you don't scale them, large values like Balance may dominate smaller values like NumOfProducts.
      #Neural Networks generally perform much better when features are on a similar scale.
#Select Columns to Scale - These are the numerical columns you want to normalize.
#Create the Scaler - scaler = MinMaxScaler() - MinMaxScaler converts values to a range between 0 and 1.
#fit() - from the training data. (Minimum Age,Maximum Age, Minimum Balance, Maximum Balance)
#transform() - Uses those values to scale the data.Before:Age = 45 & Balance = 150000, After:Age = 0.43 & Balance = 0.71
#Transform Test Data - transform() not fit_transform(), Because the model should learn scaling parameters only from the training data. This prevents data leakage.



X_train.head()

#X_train.head() displays the first 5 rows of your training dataset (X_train).
#What head() Means - which shows the first 5 rows.




X_train.shape

#X_train.shape tells you the dimensions of the training dataset.
#The output is always: (rows, columns), Ex -(8000, 11)


#Deep Learning

# model 
model = Sequential([
    layers.Dense(64, input_shape = (X_train.shape[1],), activation= 'relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1 , activation='sigmoid')
])


model.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics= ['accuracy']  
)

model.fit(
    X_train, 
    y_train,
    #class_weight= class_weight,
    epochs= 300)




#This code creates, configures, and trains an Artificial Neural Network (ANN) for churn prediction.
#Build the Neural Network
      #First Layer- layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu')
            #Dense = Fully connected layer
            #64 = 64 neurons
            #input_shape=(X_train.shape[1],) = Number of input features
            #relu = Activation function
      #Dropout Layer-layers.Dropout(0.2) -Randomly drops 20% of neurons during training. 
      #Hidden Layer 2 - layers.Dense(32, activation='relu') - Creates 32 neurons.
      #Second Dropout - layers.Dropout(0.2) - Again drops 20% of neurons during training.
      #Hidden Layer 3 - layers.Dense(16, activation='relu') - Creates 16 neurons.
      #Output Layer - layers.Dense(1, activation='sigmoid') - Since churn prediction is a binary classification problem:0 = Stay & 1 = Exit, you need only 1 output neuron.The sigmoid function produces:0.00 to 1.00
    #Neural Network Structure  
    #Input Layer (11 features)
    #  ↓
    # Dense(64) + ReLU
    #   ↓
    # Dropout(20%)
    #   ↓
    # Dense(32) + ReLU
    #   ↓
    # Dropout(20%)
    #   ↓
    # Dense(16) + ReLU
    #   ↓
    # Dense(1) + Sigmoid
    #  ↓
    # Churn Probability
#Compile the Model
      #Optimizer: Adam - Adam updates the network's weights efficiently.Think of it as the learning algorithm.
      #Loss Function - Used when there are only two classe 0 = Stay & 1 = Exit, The model tries to minimize this loss.
      #Accuracy - Shows how many predictions are correct.
#Train the Model
      #X_train - Input features: ex-Age, Balance, CreditScore, Geography_France...
      #y_train - Target: ex-0 = Stay, 1 = Exit
      #epochs=300 - One epoch means:Model sees all training data once,So The model trains 300 times through the full dataset.
        #Epoch 1/300
        # loss: 0.51
        # accuracy: 0.79
        
        # Epoch 50/300
        # loss: 0.39
        # accuracy: 0.84
        
        # Epoch 300/300
        # loss: 0.32
        # accuracy: 0.87




model.evaluate(X_test, y_test)

#model.evaluate(X_test, y_test) is used to test your trained neural network on unseen data and measure its performance.
#What happens?
   #X_test = Test features (customer information)
   #y_test = Actual churn values (0 or 1)
   #The model makes predictions on X_test and compares them with y_test.




preds = model.predict(X_test)
y_pred = (preds > 0.35).astype('int')
y_pred


#This code converts the model's prediction probabilities into 0 (No Churn) and 1 (Churn) predictions.
#Make Predictions-preds = model.predict(X_test) output - array([[0.82],[0.15],[0.67],[0.28]]), These are probabilities, not final classes.
#Apply Threshold - preds > 0.35, output - array([[ True],[False],[ True],[False]])
   #Logic:Probability > 0.35  →  True & Probability ≤ 0.35  →  False
#Convert True/False to 1/0 - .astype('int'), output - array([[1], [0],[1],[0]])
#Why 0.35 and not 0.50? - Normally:y_pred = (preds > 0.5).astype(int) is used.We are using:0.35, which is a lower threshold.
      #Using 0.35 usually:Detects more churn customers (higher Recall)
      #This is often useful in churn prediction because missing a customer who is about to leave can be costly.
#View Predictions




print(classification_report(y_test, y_pred))

#generates a detailed evaluation report comparing:
#y_test = Actual values
#y_pred = Predicted values from your model
#Precision - Of all customers predicted as churn, how many actually churned?
   #When the model predicts a customer will churn, it is correct 68% of the time.
#Recall - Of all actual churn customers, how many did the model detect?
   #he model identified 55% of all churning customers.
   #For churn prediction, Recall is very important because missing a customer who is leaving can be expensive.
#F1-Score - Combines Precision and Recall.Higher is better.
#Support = number of actual records, Means there were 407 actual churn customers in the test data.
#Class Labels - 0 = Customer Stayed & 1 = Customer Exited
   # So Class 0 metrics - show performance for customers who stayed.
   # So Class 1 metrics - show performance for customers who churned.
#Accuracy - accuracy = 0.85, 85% of all predictions were correct.



cm = tf.math.confusion_matrix(labels = y_test, predictions= y_pred)
sns.heatmap(cm , annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Actual')


#This code creates and visualizes a Confusion Matrix, which shows how well your churn model is predicting customers who stay or leave.
#What is a Confusion Matrix? - 
    #Example output:        Predicted
    #                        0    1
    #             Actual 0  1480  113
    #                    1   120  287

    #Or visually:           Predicted
    #                       Stay Exit
    #          Actual Stay  1480  113
    #          Actual Exit  120  287

    #Meaning of Each Box
    #Top Left (1480) - True Negatives (TN)
    #          Actual = Stay (0)
    #          Predicted = Stay (0)
    # Model correctly predicted customer stayed.

    #Top Right (113) - False Positives (FP)
    #          Actual = Stay (0)
    #          Predicted = Exit (1)
    # Customer stayed but model predicted churn.

    #Bottom Left (120) - False Negatives (FN)
    #          Actual = Exit (1)
    #          Predicted = Stay (0)
    # Customer churned but model missed it.
    #This is often the most important error in churn prediction.

    #Bottom Right (287) - True Positives (TP)
    #          Actual = Exit (1)
    #          Predicted = Exit (1)
    # Model correctly identified churn customers.

#Heatmap Parameters
    #annot=True - sns.heatmap(cm, annot=True) - Displays numbers inside each box.
       #Without it:Colored boxes only & With it:1480  113 & 120   287
#fmt='d'- Shows values as integers.




# I will define the model func to call it over the testing on the four algorithms
def ANN(X_train, X_test, y_train, y_test):
    model = Sequential([
        layers.Dense(64, input_shape = (X_train.shape[1],), activation= 'relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1 , activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics= ['accuracy']  
    )

    model.fit(
        X_train, 
        y_train,
        #class_weight= class_weight,
        epochs= 200)

    model.evaluate(X_test, y_test)

    preds = model.predict(X_test)
    y_pred = np.round(preds)

    print(classification_report(y_test, y_pred))

    return y_pred

#This code defines a function named ANN() so you can train and evaluate your neural network with a single function call.
#What the Function Does - def ANN(X_train, X_test, y_train, y_test):
      #The function takes:X_train → Training features, X_test → Testing features, y_train → Training labels, y_test → Testing labels
#Build the Neural Network
      #Architecture:
      #Input Layer
      # ↓
      # Dense(64)
      # ↓
      # Dropout(20%)
      # ↓
      # Dense(32)
      # ↓
      # Dropout(20%)
      # ↓
      # Dense(16)
      # ↓
      # Output(1)

#Compile the Model
      #Adam → optimizer
      #binary_crossentropy → binary classification loss
      #accuracy → performance metric
#Train the Model - The model learns from the training data for 200 epochs.
#Evaluate the Model - Tests the model on unseen data and displays:
#Make Predictions - preds = model.predict(X_test) Ex- [[0.87][0.12] [0.65][0.31]], These are probabilities.
#Convert Probability to Class - y_pred = np.round(preds), This uses a threshold of 0.5 automatically.
#Print Classification Report - print(classification_report(y_test, y_pred))
       #Displays:precision, recall, f1-score, support, accuracy
#Return Predictions - return y_pred, Allows you to save the predicted values.
 



from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=3)

rus_X_train, rus_y_train = rus.fit_resample(X_train , y_train)

ANN(rus_X_train, X_test, rus_y_train, y_test)




from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state= 3)

ros_X_train, ros_y_train = ros.fit_resample(X_train, y_train)

ANN(ros_X_train, X_test, ros_y_train, y_test)




from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state= 3)   

sm_X_train, sm_y_train = smote.fit_resample(X_train, y_train)

ANN(X_train, X_test, y_train, y_test)



from imblearn.over_sampling import BorderlineSMOTE

blsm = BorderlineSMOTE(random_state= 3)

bls_X_train, bls_y_train = blsm.fit_resample(X_train, y_train)

ANN(bls_X_train, X_test, bls_y_train, y_test)




from imblearn.combine import SMOTETomek

tom = SMOTETomek(random_state = 3)

tom_X_train, tom_y_train = tom.fit_resample(X_train, y_train)

ANN(tom_X_train, X_test, tom_y_train, y_test)





from imblearn.ensemble import BalancedRandomForestClassifier

brfc = BalancedRandomForestClassifier(n_estimators=200, random_state= 3)

brfc.fit(X_train, y_train)
brfc_pred = brfc.predict(X_test)

print(classification_report(y_test, brfc_pred))




from imblearn.ensemble import EasyEnsembleClassifier

eec = EasyEnsembleClassifier(random_state= 3)

eec.fit(X_train, y_train)
eec_pred = brfc.predict(X_test)

print(classification_report(y_test, eec_pred))
