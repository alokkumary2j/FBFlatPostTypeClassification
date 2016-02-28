
# coding: utf-8

# In[574]:

import requests
import json
import os
from nltk.corpus import stopwords
from nltk.stem import *


# In[575]:

#Login inside FB: Go to URL : https://developers.facebook.com/tools/explorer/145634995501895/
# In Short: Facebook for developers -> Tools and Support -> Graph API Explorer -> Access Token
#IMPORTANT: ACCESS_TOKEN expires very quickly(Probably, once every an hour or two)
access_token="CAACEdEose0cBADKpAoRyddc2ykaMAl0i8T6jMTqbeDG5RVSJZB0ZCfGoWFmGjN1KUdq3yZCeAfUGzsDcoPDZAZAOhT60I1pYfccd43GXDzIGaM1zPLW4O1xh8MALZC9gmmZCudjWPwj6rmYXO5wmlBQZAVxels4xgnuZA5RqBiXuo4sZCjD2WVXi4uDY0UJvJl4g2exwXyzRAvGwZDZD"
#I have Found Out Some Group Names Manually , by browsing FB
groupNames=['Flats and Flatmates (Bangalore Chapter)','Flats Without Broker in India','Flats & Flatmates : Pan India!',
            'House for Rent in Bangalore','House/Flat for Rent in Bangalore without Brokers','Bangalore House Sharing, Flats, Rentals, Rooms for sharing available'
            ,'Flats without broker -bangalore','Flats and Flatmates : PAN Bangalore !','Flats / Flatmates without brokers in India' 
           ]
groupIDs=['1463274307229712','5561469939','340254516036729','219157614840863','450055131795193','355277761216963','314309115332224',
              '391928064244514','195366117186736']
#Till, Now I am able to get POSTS posted on PUBLIC FB Groups: Out of these 9 groups, only 3 are Public:Hence have data for oly 3 groups
# Another Issue: Currently. by Deafult API only provides latest 25 POSTS for each PUBLIC group
#Hence, Below method returnGroupFeeds() will only Fetch 75 posts for each execution
baseURL="https://graph.facebook.com/"
def returnGroupFeeds():
    respJSONDataList=[]
    for grpId in groupIDs:
        url=baseURL+grpId+"/feed?access_token="+access_token
        response=requests.get(url)
        if(response.status_code!=200):
            print(response)
            print("Please Update your Access-Token : It is Expired !!!")
            return
        respJSON=response.json()
        respJSONData=respJSON["data"]
        print(len(respJSONData)," Comments Found for GRP ID : ",grpId)
        respJSONDataList.extend(respJSONData)
    return respJSONDataList


# In[576]:

jsonDataList=returnGroupFeeds()


# In[577]:

def extractPostsData(jsonDataList):
    #posts=[(jsonData.get('from'),jsonData.get('message'),{'containsPicture':'picture' in jsonData})for jsonData in jsonDataList]
    posts=[{'from':jsonData.get('from'),'message':jsonData.get('message'),'containsPicture':'picture' in jsonData} for jsonData in jsonDataList]
    return posts


# In[578]:

postsRelevantData=extractPostsData(jsonDataList)
#postsRelevantData=json.dumps(postsRelevantData)
#postsRelevantData


# In[579]:

#Uncomemnt Belwo 2 lines, when you are planning to Train the Model:However, you will still need Manual Intervention
#Ideally, Build an UI to enable Training(Training using Text Editor Sucks)
#with open('fbPostsTrainingData.model', 'w') as outfile:
#    json.dump(postData, outfile)
postsTrainingData=[]
with open('fbPostsTrainingData.model') as data_file:    
    postsTrainingData = json.load(data_file)
print(len(postsTrainingData))


# In[580]:

import nltk
#nltk.download()
from nltk.corpus import wordnet as wn
stemmer=PorterStemmer()


# In[581]:

def getSynonymsUsingWordNet(words):
    synonymSet=set()
    for word in words:
        synonymSet.add(word)
        synonyms=wn.synsets(word)
        for synonym in synonyms:
            synName=synonym.name()
            synNameparts=synName.split(".")
            synName=synNameparts[0]
            synonymSet.add(synName)
    return synonymSet


# In[582]:

def extract_entity_names(entityTree):
    entity_names = []
    #nltkNESet=('LOCATION','PERSON','ORGANIZATION','DATE','TIME','MONEY','PERCENT','FACILITY','GPE')
    nltkNESet=('LOCATION','PERSON','ORGANIZATION','DATE','TIME','MONEY','PERCENT','FACILITY','GPE')
    if(hasattr(entityTree,'label') and (entityTree.label() in nltkNESet)):
        entity_names.append(entityTree)
    return entity_names
def findAllNEChunks(text):
    for sent in nltk.sent_tokenize(text):
        neChunks=nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)))
    return neChunks
def getNECountforAPost(postDesc):
    postDesc=postDesc.replace("\n",".")
    neChunks=findAllNEChunks(postDesc)
    entityNames = []
    for neChunk in neChunks:
        entityNames.extend(extract_entity_names(neChunk))
    #return entityNames
    return len(entityNames)
def getAmenitiesCount(postDesc):
    #List of Amenities - Courtesy : CommonFloor+FB Posts :-)
    amenitiesSet=('garden','recreat','power','backup','clubhous','harvest','commun','swim',
                  'gym','gymnasium','indoor','intercom','tile','park','vitrifi','cupboard','kitchen',
                  'live','din','balconi','wardrob','granit','floor','wifi','tv','maid','servant',
                  'electr','badminton','tenni','basketball','basket','modular','microwav','groceri','store',
                  'oven','tataski','wash','aquaguard','play','fridg','attach','restroom','bathroom','separate',
                  'clubhous','intercom','gate','up','wardrob','cupboard','filter','hall','sofa','cupboard','tata'
                 'purifi','airport','busstand','busstop','bu','rack','room','tabl','furnitur','dine','televis',
                  'cctv','geyser','wi-fi','machin','secur','internet')
    postDesc=postDesc.lower()
    words=nltk.word_tokenize(postDesc)
    wordMembership=[stemmer.stem(word) in amenitiesSet for word in words]
    #requires Scaling[POSTS can be of totally Different Size and size does impact the frequency here for the given class say , "SELL"]
    return sum(wordMembership)/len(words)


# In[583]:

#Used this module to get the  STEMS: ONE TIME EFFORT: SHOULDN'T RECOMPUTE
#tenantPhraseList=[('looking','apartment'),('looking','flat'),('looking','house'),('leads','please'),('any','leads')]
#tenantPhraseList.extend([('new','delhi'),('new','bangalore'),('new','chennai'),('new','pune'),('new','mumbai')])
#ownerPhraseList=[('looking','flatmate'),('looking','pg'),('looking','male'),('looking','female'),('looking','boy'),('looking','girl')]
#ownerPhraseList.extend([('available','apartment'),('available','flat'),('available','house')])
#ownerPhraseList.extend([('flatmate','required'),('need','flatmate'),('flat','mate')])
#ownerPhraseList.extend([('good','access'),('nice','locality'),('shared','accommodation'),('walking','distance')])
#stems=[stemmer.stem(w[1]) for w in ownerPhraseList]


# In[584]:

def getStemmedBigrams(postDescription):
    postDescription=postDescription.replace("#"," ")
    commentWords=nltk.word_tokenize(postDescription)
    stopwordList=stopwords.words('english')
    stopwordList.extend([',',':','','(',')','&','#','.','!','!!','/','bhk','hk','rk','1bhk','1/2bhk','2bhk','2/3bhk','3bhk','3/4bhk','4bhk'])
    stopwordList.extend(range(9))
    content = [comment.lower() for comment in commentWords if comment.lower() not in stopwordList]
    stemmedContent=[stemmer.stem(cont) for cont in content]
    contentBigrams=list(nltk.bigrams(stemmedContent))
    return contentBigrams


# In[585]:

def getTenantWordCount(postDescription):
    postDescription=postDescription.lower()
    tenantWordList=["budget"]
    #tenantWordSet=getSynonymsUsingWordNet(tenantWordList)
    count=0
    for tenantWord in tenantWordList:
        count+=postDescription.count(tenantWord)
    return count#Doesn't require Scaling, bcoz Size of a post doesn't REALLY increase the probability of its occurence
def getOwnerWordCount(postDescription) :
    postDescription=postDescription.lower()
    ownersWordList=["flatmate","rent","deposit"]#Problem is "rent" and "deposit" if come with "budget"It's on BUY side
    #ownersWordSet=getSynonymsUsingWordNet(ownersWordList)
    count=0
    for ownersWord in ownersWordList:
        count+=postDescription.count(ownersWord)
    #Doesn't require Scaling, bcoz Size of a post doesn't REALLY increase the probability of its occurence
        #people will usually mention it once, if they do
    return count
#this method return 1(Tenant/BUYER);-1(OWNER/SELLER),0:Neutral
def getAggregateOwnerTenantCat(postDescription):
    postDescription=postDescription.lower()
    tenantWordList=["budget"]
    ownersWordList=["flatmate","rent","deposit"]#Problem is "rent" and "deposit" if come with "budget"It's on BUY side
    for tenantWord in tenantWordList:
        if postDescription.count(tenantWord)>0:
            return 1
    for ownersWord in ownersWordList:
        if postDescription.count(ownersWord)>0:
            return -1
    return 0
#Below method counts the frequency of matching Stemmed bigrams to Tenant Stemmed Bigrams
def getTenantPhraseCount(postDescription):
    tenantPhraseListLev1=[('look','apart'),('look','flat'),('look','near'),('look','around'),('look','pg'),
                          ('look','hous'),('look','accommod'),('lead','pleas'),('ani','lead')]
    #tenantPhraseListLev1 contains Stemmed Bigrams whose occurence exhibits very high probability of the POST posted by Tenant
    tenantPhraseList=[('seek','apart'),('seek','flat'),('seek','accommod'),('seek','hous'),('lead','pleas'),
                      ('ani','lead')]
    tenantPhraseList.extend([('new','delhi'),('new','bangalor'),('new','chennai'),('new','pune'),('new','mumbai')])#Intuition is sentences like : "I am new to Bangalore and looking for ..."
    count=0
    tenantBigrams=getStemmedBigrams(postDescription)
    for (w1,w2) in tenantBigrams:
        if(w1,w2) in tenantPhraseListLev1:
           count = count+3
        elif(w1,w2) in tenantPhraseList:
           count = count+1
    return count#No need to Scale:Length of post doesn't REALLY changes the fequency count
#Below method counts the frequency of matching Stemmed bigrams to Owner's Stemmed Bigrams
def getOwnerPhraseCount(postDescription) :
    ownerPhraseList=[('look','flatmat'),('look','male'),('look','femal'),('look','boy'),
                     ('look','girl'),('pg','girl'),('pg','boy'),('pg','male'),('pg','femal')]
    ownerPhraseList.extend([('avail','apart'),('avail','flat'),('avail','hous'),('avail','room'),('avail','sale'),('avail','rent')])
    ownerPhraseList.extend([('apart','avail'),('flat','avail'),('hous','avail')])
    ownerPhraseList.extend([('flatmat','requir'),('need','flatmat'),('need','roommat')
                            ,('flat','mate'),('share','accommod')])
    #PG folks will say "Looking for a Shared Accomodation"
    ownerPhraseList2=[('good','access'),('nice','local'),('walk','distanc')]#It might Need Scaling(Highly Corrlated - might co-occuer,Problem for short POSTS) 
    #Owners do tend to describe in terms of- "nice locality","good accessibility","walking distance from Landmark X"
    count=0
    ownerBigrams=getStemmedBigrams(postDescription)
    for (w1,w2) in ownerBigrams:
        if(w1,w2) in ownerPhraseList:
           count=count+1
    for (w1,w2) in ownerBigrams:
        if(w1,w2) in ownerPhraseList2:
           count=count+1 #This Counting for the time being is not scaled: Needs Some Smart Scaling
    #No need to scale, Size doesn't really impact the frequency of these terms[ownerPhrase2 has exception]
        #Ideally, there should be some scaling on ownerPharseList2 items occurence:I am ignoring for the timebeing
    return count
def doesContainPG(postDescription):
    postDescription=postDescription.replace("\n"," ")
    postDescription=postDescription.lower()
    listOfPGTerms=["pg","paying guest"]
    for pgterm in listOfPGTerms:
        if postDescription.find(pgterm) >= 0:
            return True
    return False
def generateFeatuesOnAFlatRelatedPost(postedData):
    postDescription=postedData['message']
    features = {}
    #features["count_NEs"]=getNECountforAPost(postDescription) #Done; Seems to be a Bad Feature
    features["containsImage"]=postedData['containsPicture'] #Done
    #features["count_tenant_words"] = getTenantWordCount(postDescription)#Onlyfew Words are Non-ambiguous
    #features["count_owner_words"] = getOwnerWordCount(postDescription) #Only Few Words are are Non-Ambiguous
    features["aggregatedOwnerTenantCat"]=getAggregateOwnerTenantCat(postDescription)
    features["count_tenant_Phrases"] = getTenantPhraseCount(postDescription) #Done
    features["count_owner_Phrases"] = getOwnerPhraseCount(postDescription) #Done
    features["count_Amenities"]=getAmenitiesCount(postDescription) #Done
    features["isPGRelated"]=doesContainPG(postDescription)
    #doesContainPG is a Good feature for specific cases;However I doubt that Traing Data contains PG :
    #So, the Model at the moment mayn't be leveraging ths feature
    #features["NoOfPOstsMadebyUser"]=getNumOfPostsBySameUser(postData[0]) #Good feature, but don't have sufficient Data
    #features["description_size"]=len(postDescription) #Turned Out to be a Bad Feature with other Combinations
    return features


# In[586]:

postTypeModel_train_set=[(generateFeatuesOnAFlatRelatedPost(postTrainData),postTrainData['typeOfPost']) for postTrainData in postsTrainingData]
len(postTypeModel_train_set)
#train_set,test_set=postTypeModel_train_set[:30],postTypeModel_train_set[30:]#Trained on 30 posts, Will evaluate on 15 posts
#train_set,test_set=postTypeModel_train_set[:40],postTypeModel_train_set[5:]#Trained on 40 posts, Will evaluate on 5 posts:Since I have very less traing dataset and taht too is skewed
train_set,test_set=postTypeModel_train_set[:44],postTypeModel_train_set[44:]#Trained on 44 posts, Will evaluate on 1 posts:Since I have very less traing dataset and taht too is skewed
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


# In[587]:

#ERROR ANALYSIS ON THE TRAINING DATASET
errors = []
for postTrainData in postsTrainingData:
    tag=postTrainData['typeOfPost']
    guess = classifier.classify(generateFeatuesOnAFlatRelatedPost(postTrainData))
    if guess != tag:
        errors.append( ("ACTUAL : "+tag, "GUESSED : "+guess, postTrainData) )
print(errors)


# In[588]:

#Run MODEL on DYNAMIC/NEW DATASETS
modelResults=[]
for newPost in postsRelevantData:
    if newPost is None or newPost['message'] is None:
        print("<Data Issue>Will Skip this Post : ",newPost,"</Data Issue>")
        continue
    newPost['message']=newPost['message'].replace("\n"," ")
    #<IMPORTANT>SHOULD HAVE A SPELL CORRECTOR MODULE HERE..BEFORE RUNNING BAYESIAN MODEL</IMPORTANT>
    guess=classifier.classify(generateFeatuesOnAFlatRelatedPost(newPost))
    modelResults.append((guess,newPost['message']))
print(modelResults)

