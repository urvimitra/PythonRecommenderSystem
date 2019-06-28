import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])
df.head()
movie_titles=pd.read_csv('Movie_Titles')
movie_titles.head()
df=pd.merge(df,movie_titles,on='item_id')
df.describe()
ratings=pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
ratings['no_of_ratings']=df.groupby('title')['rating'].count()
ratings.head()
import matplotlib.pyplot as plt
%matplotlib inline
ratings['rating'].hist(bins=50)
ratings['no_of_ratings'].hist(bins=60)
import seaborn as sns
sns.jointplot(x='rating',y='no_of_ratings',data='ratings')
movie_matrix=pd.pivot_table(index='user_id',columns='title',values='rating')
movie_matrix.head()
ratings.sort_values('no_of_ratings',ascending=False).head(10)
afo_user_rating=movie_matrix['AirForceOne']
contact_user_rating=movie_matrix['Contact']
afo_user_rating.head()
contact_user_rating.head()
similartoafo=movie_matrix.corrwith(afo_user_rating)
similartoafo.head()
similartocontact=movie_matrix.corrwith(contact_user_rating)
similartocontact.head()
corr_contact=pd.DataFrame(similartocontact,columns=['Correlation'])
corr_contact.dropna(inplace=True)
corr_contact.head()
corr_afo=pd.DataFrame(similartoafo,columns=['correlation'])
corr_afo.dropna(inplace='True')
corr_afo.head()
corr_afo=corr_afo.join(ratings['no_of_ratings'])
corr_contact=corr_contact.join(ratings['no_of_ratings])
corr_afo.head()
corr_contact.head()
corr_afo[corr_afo['no_of_ratings']>100].sort_values(by='Correlation',ascending=False).head(10)
corr_contact[corr_contact['no_of_ratings'>100].sort_values(by='Correlation',ascending=False).head(10)
