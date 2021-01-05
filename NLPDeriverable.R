#_______________________________________________________________________
#                            NLP deliverable                            
#              Madrid's airbnb reviews, sentiment analysis        
#                        Author: Ana Marin Estañ                        
#_______________________________________________________________________


#__ 1. Check prerequesites _____________________________________________
# Fast option: just works with a random sample of the dataset
# execution time is significantly smaller
sampleReviews = TRUE
# Check if packages are installed and added to library
packages <- c('rstudioapi', 
              'DT', 
              'tm',
              'ggplot2',
              'textcat',
              'wordcloud',
              'RWeka',
              'reshape2', 
              'tidyverse', 
              'tidytext', 
              'stringr',
              'dplyr', 
              'topicmodels')

install.packages(setdiff(packages, rownames(installed.packages())))  
rm(packages)

library(rstudioapi)
library(DT)
library(tm)
library(ggplot2)
library(textcat)
library(wordcloud)
library(RWeka)
library(reshape2)
library(tidyverse)
library(tidytext)
library(stringr)
library(dplyr)
library(topicmodels)


#Set work dir
setwd(dirname(getActiveDocumentContext()$path ))
cat('Working directory:', getwd())


#__ 2. Import data and basic exploration _______________________________

#Import data
reviews <- read_csv('data/reviews.csv')

#Show data head in plot 
datatable(head(reviews, 10))

#Show number of reviews:
cat('Number of reviews:', nrow(reviews), '\nColumn names:', names(reviews))


#__ 3. Basic data preprocessing ________________________________________

#Limit data to 1% random reviews sample
if(sampleReviews) {
  reviewsPrep = reviews[sample(nrow(reviews), as.integer(nrow(reviews)/100)),]
} else {
  reviewsPrep = reviews
}

#Detect review language
reviewsPrep$language <- textcat(reviewsPrep$comments)
reviewsPrep$language <- as.factor(reviewsPrep$language)

#Group reviews by language and show 10 most frequent languages 
languageCount = count(reviewsPrep, language)
languageCount = languageCount[order(languageCount$n, decreasing = T),]
languageCount$percentage <- (languageCount$n/nrow(reviewsPrep))*100
  
# Keep reviews written in English and Spanish (two most frequent languages)
# in different data frames
reviewsEn = reviewsPrep[c(reviewsPrep$language == 'english'),]
reviewsEs = reviewsPrep[c(reviewsPrep$language == 'spanish'),]

# Remove empty reviews
reviewsEn = reviewsEn[!(is.na(reviewsEn$comments) | reviewsEn$comments==''), ]
reviewsEs = reviewsEs[!(is.na(reviewsEs$comments) | reviewsEs$comments==''), ]

#__ 4. Corpora creation, inspection and transformation _________________
# EN
# Create and process corpus
corpusEn = VCorpus(VectorSource(reviewsEn$comments))
cat('English corpus length:', length(corpusEn), 'entries')

summary(corpusEn[1:5]); meta(corpusEn[[1]]); content(corpusEn[[1]])

stopwordsEn = c(stopwords(),'airbnb', 'madrid','stay')
corpusEnT <- tm_map(corpusEn, content_transformer(tolower))
corpusEnT <- tm_map(corpusEnT, content_transformer(removeNumbers))
corpusEnT <- tm_map(corpusEnT, content_transformer(removeWords), stopwordsEn)
corpusEnT <- tm_map(corpusEnT, content_transformer(removePunctuation))
corpusEnT <- tm_map(corpusEnT, content_transformer(stemDocument))
corpusEnT <- tm_map(corpusEnT, content_transformer(stripWhitespace))

# Compare documents
corpusEn[['100']][['content']]; corpusEnT[['100']][['content']]

# ES
# Creation and inspection
corpusEs = VCorpus(VectorSource(reviewsEs$comments))
cat('Spanish corpus length:', length(corpusEs), 'entries')
summary(corpusEs[1:5]); meta(corpusEs[[1]]); content(corpusEs[[1]])

# Transformation
stopwordsEs = c(stopwords('spanish'), 'madrid', 'airbnb')
corpusEsT <- tm_map(corpusEs, content_transformer(tolower))
corpusEsT <- tm_map(corpusEsT, content_transformer(removeNumbers))
corpusEsT <- tm_map(corpusEsT, content_transformer(removeWords), stopwordsEs)
corpusEsT <- tm_map(corpusEsT, content_transformer(removePunctuation))
corpusEsT <- tm_map(corpusEsT, content_transformer(stemDocument))
corpusEsT <- tm_map(corpusEsT, content_transformer(stripWhitespace))

# Compare documents
corpusEs[['50']][['content']]; corpusEsT[['50']][['content']]


#__ 5. Create TDM ______________________________________________________
# create TDM with TF-IDF weighting
tdmEn = TermDocumentMatrix(corpusEnT, control = list(weighting = weightTfIdf))
tdmEn

tdmEs = TermDocumentMatrix(corpusEsT, control = list(weighting = weightTfIdf))
tdmEs


#__ 6. Basic text analysis _____________________________________________
if(sampleReviews) {
  maxWords = 100
  lowFreqEn = 1000
  lowFreqEs = 500
} else {
  maxWords = 10000
  lowFreqEn = 100000
  lowFreqEs = 50000
}
# Language analysis
languageCount[1:10,]
p <- ggplot(data=select(languageCount[1:10, ], language, n), aes(x=reorder(language, -n), y=n)) +
  geom_bar(stat='identity', fill='#2F3456')
p + xlab('Reviews') +
  ylab('Detected language') +
  coord_flip()

#TF-IDF Zipf's law
freqEn=rowSums(as.matrix(tdmEn))
freqEs=rowSums(as.matrix(tdmEs))
par(mfrow=c(2,1), mar = c(2, 2, 2, 2))
plot(sort(freqEn, decreasing = TRUE), col='#2F3456', xlab='TF-IDF-based rank', ylab = 'TF-IDF', main='Word TF-IDF frequencies (English)')
plot(sort(freqEs, decreasing = TRUE), col='#E77161', xlab='TF-IDF-based rank', ylab = 'TF-IDF', main='Word TF-IDF frequencies (Spanish)')
print('10 most relevant words in English:'); tail(sort(freqEn),n=10)
print('10 most relevant words in Spanish:'); tail(sort(freqEs),n=10)

#Word Clouds
par(mfrow=c(1,1))

unigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
tdmEn.unigram = TermDocumentMatrix(corpusEnT,
                                control = list (weighting = weightTfIdf,
                                                tokenize = unigramTokenizer))
tdmEs.unigram = TermDocumentMatrix(corpusEsT,
                                   control = list (weighting = weightTfIdf,
                                                   tokenize = unigramTokenizer))

# Plot English Wordcloud
freqEn = sort(rowSums(as.matrix(tdmEn.unigram)),decreasing = TRUE)
freqEn.df = data.frame(word=names(freqEn), freq=freqEn)
layout(matrix(c(1, 2), nrow=2), heights=c(1, 10))
par(mar=rep(0, 4))
plot.new()
text(x=0.5, y=0.5, "English wordcloud", cex=1.5,font=2)
wordcloud(freqEn.df$word,freqEn.df$freq,max.words=maxWords,random.order = FALSE, colors=brewer.pal(6,'Paired'))

# Plot Spanish Wordcloud
freqEs = sort(rowSums(as.matrix(tdmEs.unigram)),decreasing = TRUE)
freqEs.df = data.frame(word=names(freqEs), freq=freqEs)
layout(matrix(c(1, 2), nrow=2), heights=c(1, 10))
par(mar=rep(0, 4))
plot.new()
text(x=0.5, y=0.5, "Spanish wordcloud", cex=1.5,font=2)
wordcloud(freqEs.df$word,freqEs.df$freq,max.words=maxWords,random.order = FALSE, colors=brewer.pal(6,'Paired'))

#Word frequencies and associations
dtmEn <- DocumentTermMatrix(corpusEnT)
dtmEs <- DocumentTermMatrix(corpusEsT)

freqTermEn = findFreqTerms(dtmEn, lowfreq = lowFreqEn)
assocEn = findAssocs(dtmEn, terms = freqTermEn, corlimit = 0.15)	
freqTermEn
assocEn

freqTermEs = findFreqTerms(dtmEs, lowfreq = lowFreqEs)
assocEs = findAssocs(dtmEs, terms = freqTermEs, corlimit = 0.15)	
freqTermEs
assocEs

#__ 7. Sentiment analysis ______________________________________________
#Get sentiments data frame
dtmEnTidy <- tidy(dtmEn)
sentimentsEn <- dtmEnTidy %>%
  inner_join(get_sentiments('bing'), by = c(term = 'word'))

positiveWordsEs <- read_csv('es_lexicons/positive_words_es.txt', col_names = "term") %>%
  mutate(sentiment = "positive")
negativeWordsEs <- read_csv('es_lexicons/negative_words_es.txt', col_names = "term") %>%
  mutate(sentiment = "negative")
sentimentWordsEs <- bind_rows(positiveWordsEs, negativeWordsEs)

dtmEsTidy <- tidy(dtmEs)
sentimentsEs <- dtmEsTidy %>%
  inner_join(sentimentWordsEs) %>%
  count(document, term, sentiment, sort = TRUE) %>%
  ungroup()
names(sentimentsEs)[4] <- 'count'

if(sampleReviews) {
  termFreq = 40
} else {
  termFreq = 1500
}

# EN
#General trends
sentimentsEn %>%
  count(sentiment, wt = count) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  arrange(-sentiment)

# Find most positive reviews
posRevs <- sentimentsEn %>%
  count(document, sentiment, wt = count) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  arrange(-sentiment) %>%
  head()
# Inspect most positive doc
sentimentsEn %>%
  filter(document == posRevs$document[1]) %>%
  arrange(-count)
corpusEn[[posRevs$document[1]]][['content']]  

# Find most negative reviews
negRevs <- sentimentsEn %>%
  count(document, sentiment, wt = count) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  arrange(sentiment)
# Inspect most negative doc
sentimentsEn %>%
  filter(document == negRevs$document[1]) %>%
  arrange(-count)
corpusEn[[negRevs$document[1]]][['content']]

# Most sentiment-contributive words 
sentimentsEn %>%
  count(sentiment, term, wt = count) %>%
  filter(n >= termFreq) %>%
  mutate(n = ifelse(sentiment == 'negative', -n, n)) %>%
  mutate(term = reorder(term, n)) %>%
  ggplot(aes(term, n, fill = sentiment)) +
  geom_bar(stat = 'identity') +
  coord_flip() +
  geom_col() +
  theme() +
  labs(x = 'Term', y = 'Contribution to sentiment',  title = 'English')

# ES
# General trends
sentimentsEs %>%
  count(sentiment, wt = count) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  arrange(-sentiment)

# Find most positive reviews
posRevs <- sentimentsEs %>%
  count(document, sentiment, wt = count) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  arrange(-sentiment)
# Inspect most positive doc
sentimentsEs %>%
  filter(document == posRevs$document[1]) %>%
  arrange(-count)
corpusEs[[posRevs$document[1]]][['content']]

# Find most negative reviews
negRevs <- sentimentsEs %>%
  count(document, sentiment, wt = count) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  arrange(sentiment)
# Inspect most negative doc
sentimentsEs %>%
  filter(document == negRevs$document[1]) %>%
  arrange(-count)
corpusEs[[negRevs$document[1]]][['content']]
# Most sentiment-contributive words 
sentimentsEs %>%
  count(sentiment, term, wt = count) %>%
  filter(n >= termFreq) %>%
  mutate(n = ifelse(sentiment == 'negative', -n, n)) %>%
  mutate(term = reorder(term, n)) %>%
  ggplot(aes(term, n, fill = sentiment)) +
  geom_bar(stat = 'identity') +
  coord_flip() +
  geom_col() +
  theme() +
  labs(x = "Term", y = 'Contribution to sentiment',  title = "Spanish")

