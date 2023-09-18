#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.text import text_to_word_sequence
import nltk
from nltk.corpus import stopwords
import pickle
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# In[71]:


df = pd.read_csv("data/lenova.csv", encoding='latin-1', header=None, usecols=[4,5,6])
df.head(5)


# In[72]:


#df = df[[4,5,6]].copy()
df.columns = ['rating', 'reviewTitle', 'reviews']
df.head(5)


# In[73]:


df.dtypes


# In[74]:


df['sentiment'] = df['rating'].apply(lambda rate: 1 if (rate>3) else 0)
print(df.head(5))


# In[75]:


df['sentiment'].value_counts()


# In[76]:


9001/17599


# ### Balance the data

# In[77]:


dfPos = df[df['sentiment'] == 1]
dfPos.shape


# In[78]:


dfNeg = df[df['sentiment'] == 0]
dfNeg.shape


# In[79]:


dfPosDownsize = dfPos.sample(dfNeg.shape[0])
dfPosDownsize.shape


# In[80]:


dfBalance = pd.concat([dfPosDownsize,dfNeg])
dfBalance.shape


# In[81]:


dfBalance['sentiment'].value_counts()


# In[82]:


dfFinal = dfBalance.copy()
print(dfFinal.head(10))


# In[48]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[83]:


reviews = df['reviews'].values.tolist()
labels = df['sentiment'].tolist()
print(reviews[:2])
print(labels[:2])


# In[84]:


X_train, X_temp, y_train, y_temp = train_test_split(reviews, labels, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# In[85]:


#get_ipython().system('pip install transformers')


# In[86]:


from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import torch
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[87]:


device


# In[88]:


#!nvidia-smi


# In[89]:


X_train[0]


# In[90]:


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# In[91]:


tokenizer([X_train[0]], truncation=True,
                            padding=True, max_length=128)


# In[25]:


# pip install GPUtil


# In[26]:


# import GPUtil
# GPUtil.getAvailable()


# In[92]:


train_encodings = tokenizer(X_train,
                            truncation=True,
                            padding=True)
val_encodings = tokenizer(X_val,
                            truncation=True,
                            padding=True)
test_encodings = tokenizer(X_test,
                            truncation=True,
                            padding=True)


# In[93]:


train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    y_val
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))

test_dataset_path = "test_dataset.tfrecord"
tf.data.experimental.save(test_dataset, test_dataset_path)

# In[94]:


train_dataset


# In[95]:


model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=2)


# In[96]:


model.summary()


# In[ ]:
#early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1,mode='min')

checkpoint = ModelCheckpoint("bert.h5", save_best_only=True, mode='max', monitor='val_accuracy', save_weights_only=True, verbose = 1)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy','crossentropy'])
history=model.fit(train_dataset.shuffle(100).batch(16),
          epochs=20,
          batch_size=16,
          callbacks = [checkpoint],
          validation_data=val_dataset.shuffle(100).batch(16))

model.save_pretrained('fine_tuned_bert')

# In[ ]:


# Print the evaluation results
#print("Evaluation Loss:", evaluation_result[0])
#print("Evaluation Accuracy:", evaluation_result[1])

# Save the history object to a file
with open('history.pkl', 'wb') as file:
    pickle.dump(history, file)


# In[ ]:


# Make predictions on test set using the loaded model
test_predictions = model.predict(test_dataset)
test_predictions = np.argmax(test_predictions.logits, axis=1)

print('test_predictions length: ',len(test_predictions))
print('test_predictions: ',test_predictions)
print('==========================================================================')
print('test length: ',len(y_test))

# Generate classification report
print(classification_report(y_test, test_predictions))


# In[ ]:


# Create a DataFrame with the predicted labels
dfPred = pd.DataFrame({'Predicted': test_predictions})

# Save the DataFrame to a CSV file
dfPred.to_csv('predictions1.csv', index=False)

# Create a DataFrame with the original labels
dfTrue = pd.DataFrame({'true': y_test})

# Save the original DataFrame to a CSV file
dfTrue.to_csv('orig.csv', index=False)

# Create a DataFrame with the original data
dfXTest = pd.DataFrame({'X': X_test})

# Save the original data DataFrame to a CSV file
dfXTest.to_csv('XTest.csv', index=False)

# In[72]:

dfPredicted = pd.DataFrame(columns=['Text', 'Original Sentiment', 'Predicted Sentiment'])

test_sentence = "This is a super phone I have used for one month no problem was happened it is a super and Wonder ful phone.......Happy"

predict_input = tokenizer.encode(test_sentence,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")

tf_output = model.predict(predict_input)[0]


tf_prediction = tf.nn.softmax(tf_output, axis=1)
labels = ['Negative','Positive']
label = tf.argmax(tf_prediction, axis=1)
label = label.numpy()
predData = {'Text': test_sentence, 'Original Sentiment': 'Positive', 'Predicted Sentiment': labels[label[0]]}
new_df = pd.DataFrame([predData])
dfPredicted = pd.concat([dfPredicted,new_df], ignore_index=True)


# In[73]:

test_sentence = "I don't understand why manufacturer boast only 2750mah battery.Absolutely rubbish mobile every mobile having quick charge function but it's having quick discharge function."

predict_input = tokenizer.encode(test_sentence,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")

tf_output = model.predict(predict_input)[0]


tf_prediction = tf.nn.softmax(tf_output, axis=1)
labels = ['Negative','Positive']
label = tf.argmax(tf_prediction, axis=1)
label = label.numpy()
predData = {'Text': test_sentence, 'Original Sentiment': 'Negative', 'Predicted Sentiment': labels[label[0]]}
new_df = pd.DataFrame([predData])
dfPredicted = pd.concat([dfPredicted,new_df], ignore_index=True)


# In[74]:
test_sentence = "Must Read this full review...  I'm switched from infocus M2 3G to this Lenovo vibe k5 plus... @6749 on BBD sale. After one day use , I write my review From the package I got mobile, battery, charging adapter, data cable, plastic back case, screen protection film but no ear phones. *Design is looking good and handy. *Camera back 13 mp but not better than my infocus M2(8mp) , both are almost equal. *camera front is very clear to take selfies. *touch is very good to use and display amazing. *Dolby speakers , sound is better than normal.*phone is heating very rarely.  *comes to multitasking, 3GB RAM works fine , not hanged till now. *Battery backup is not good as excepted , but OK for normal use. I chooses this mobile for this price tag of Rs.6749 at BBD sale , I would recommend at price tag of 8.5k search for other Mobiles with high battery back up. If u don't want much battery back up, then go for this mobile. Wait wait wait ! The above given review is only after one day use , now I'm editing this review after one month heavy usage of lenevo vibe k5 plus... This is worst phone I've ever used , very much irritating to use , heating , hanging and everything was happening...  Not able to use continuously for 1hour , after continuously using one hour , it heats and hangs on every  action, even calls are not able to attend on that time , only ringtone is ringing but screen is in black... Conclusion : this phone specs only on label 3GB ram , 1.5GHz Octacore,,, this is processing like 512MB ram . I'm saled this bricky mobile to low price , and bought Infocus bingo 50+ and this review also writing on my new Infocus mobile... Guys I'm shared my experience with Lenevo vibe k5 plus , now u decide u wanna buy it not...  Thank you !"

predict_input = tokenizer.encode(test_sentence,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")

tf_output = model.predict(predict_input)[0]


tf_prediction = tf.nn.softmax(tf_output, axis=1)
labels = ['Negative','Positive']
label = tf.argmax(tf_prediction, axis=1)
label = label.numpy()
predData = {'Text': test_sentence, 'Original Sentiment': 'Negative', 'Predicted Sentiment': labels[label[0]]}
new_df = pd.DataFrame([predData])
dfPredicted = pd.concat([dfPredicted,new_df], ignore_index=True)


print("===============================================================")
print(dfPredicted)


model.save_pretrained('fine_tuned_bert')
loaded_model = TFDistilBertForSequenceClassification.from_pretrained('fine_tuned_bert')
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loaded_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy','crossentropy'])
tdataset = tf.data.experimental.load(test_dataset_path)
evaluation_result = loaded_model.evaluate(tdataset.batch(16))
print("Evaluation Loss:", evaluation_result[0])
print("Evaluation Accuracy:", evaluation_result[1])
