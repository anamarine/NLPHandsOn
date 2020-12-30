#_______________________________________________________________________
#                            NLP deliverable                            
#              Madrid's airbnb reviews, sentiment analysis        
#                        Author: Ana Marin Estañ                        
#_______________________________________________________________________


#__ 1. Check prerequesites _____________________________________________

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

#Limit data to 2500 random reviews
reviewsPrep = reviews[sample(nrow(reviews), 2500),]

#Detect review language
reviewsPrep$language <- textcat(reviewsPrep$comments)
reviewsPrep$language <- as.factor(reviewsPrep$language)

#Group reviews by language and show 10 most frequent languages 
languageCount = count(reviewsPrep, language)
languageCount = languageCount[order(languageCount$n, decreasing = T),]
languageCount$percentage <- (languageCount$n/2000)*100
languageCount[1:10,]

# Keep reviews written in English and Spanish (two most frequent languages)
# in different data frames

reviewsEn = reviewsPrep[c(reviewsPrep$language == 'english'),]
reviewsEs = reviewsPrep[c(reviewsPrep$language == 'spanish'),]
# Remove empty reviews
reviewsEn = reviewsEn[!(is.na(reviewsEn$comments) | reviewsEn$comments==''), ]
reviewsEs = reviewsEs[!(is.na(reviewsEs$comments) | reviewsEs$comments==''), ]

#__ 4. Load and inspect Corpus _________________________________________
# Process English corpus
corpusEn = Corpus(VectorSource(reviewsEn$comments))
cat('English corpus length:', length(corpusEn), 'entries')
if(length(corpusEn) > 1000)
  corpusEn=corpusEn[1:1000]
  cat('Limit English corpus length to:', length(corpusEn), 'entries')
summary(corpusEn[1:5])
inspect(corpusEn[[1]])
meta(corpusEn[[1]])
content(corpusEn[[1]])

# Process Spanish corpus
corpusEs = Corpus(VectorSource(reviewsEs$comments))
cat('Spanish corpus length:', length(corpusEs), 'entries')
if(length(corpusEs) > 1000)
  corpusEs=corpusEs[1:1000]
  cat('Limit Spanish corpus length to:', length(corpusEs), 'entries')

content(corpusEs[[1]])


#__ 5. Create and transform DTM ________________________________________

stopwordsEn = c(stopwords(),'airbnb', 'madrid','stay')
corpusEnT <- tm_map(corpusEn, tolower)
corpusEnT <- tm_map(corpusEnT, removeNumbers)
corpusEnT <- tm_map(corpusEnT, removeWords, stopwordsEn)
corpusEnT <- tm_map(corpusEnT, stemDocument)
corpusEnT <- tm_map(corpusEnT, removePunctuation)
corpusEnT <- tm_map(corpusEnT, stripWhitespace)

stopwordsEs = c(stopwords('spanish'), 'madrid', 'airbnb')
corpusEsT <- tm_map(corpusEs, tolower)
corpusEsT <- tm_map(corpusEsT, removeNumbers)
corpusEsT <- tm_map(corpusEsT, removeWords, stopwordsEs)
corpusEsT <- tm_map(corpusEsT, stemDocument)
corpusEsT <- tm_map(corpusEsT, removePunctuation)
corpusEsT <- tm_map(corpusEsT, stripWhitespace)


# Compare transdormed documents
corpusEn[['100']][['content']]; corpusEnT[['100']][['content']]
corpusEs[['50']][['content']]; corpusEsT[['50']][['content']]


#__ 6. Create TDM ______________________________________________________
# create TDM with TF-IDF weighting
tdmEn = TermDocumentMatrix(corpusEnT, control = list(weighting = weightTfIdf))
tdmEn

tdmEs = TermDocumentMatrix(corpusEsT, control = list(weighting = weightTfIdf))
tdmEs


#__ 7. Basic text analysis _____________________________________________

#TF-IDF word frequencies
freqEn=rowSums(as.matrix(tdmEn))
freqEs=rowSums(as.matrix(tdmEs))
par(mfrow=c(2,1), mar = c(2, 2, 2, 2))
plot(sort(freqEn, decreasing = TRUE),col='blue',main='Word TF-IDF frequencies (English)', xlab='TF-IDF-based rank', ylab = 'TF-IDF')
plot(sort(freqEs, decreasing = TRUE),col='red',main='Word TF-IDF frequencies (Spanish)', xlab='TF-IDF-based rank', ylab = 'TF-IDF')
print('10 most relevant words in English:'); tail(sort(freqEn),n=10)
print('10 most relevant words in Spanish:'); tail(sort(freqEs),n=10)

#Word Clouds
palEn=brewer.pal(4,'Blues')
palEs=brewer.pal(4,'Reds')
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
layout(matrix(c(1, 2), nrow=2), heights=c(1, 4))
par(mar=rep(0, 4))
plot.new()
text(x=0.5, y=0.5, "English wordcloud", cex=2,font=2)
wordcloud(freqEn.df$word,freqEn.df$freq,max.words=100,random.order = FALSE, colors=palEn)

# Plot Spanish Wordcloud
freqEs = sort(rowSums(as.matrix(tdmEs.unigram)),decreasing = TRUE)
freqEs.df = data.frame(word=names(freqEs), freq=freqEs)
plot.new()
text(x=0.5, y=0.5, "Spanish wordcloud", cex=2,font=2)
wordcloud(freqEs.df$word,freqEs.df$freq,max.words=100,random.order = FALSE, colors=palEs)

#__ 8. Sentiment analysis ______________________________________________
#Get sentiments data frame
dtmEn <- DocumentTermMatrix(corpusEnT)
dtmEnTidy <- tidy(dtmEn)
sentimentsEn <- dtmEnTidy %>%
  inner_join(get_sentiments('bing'), by = c(term = 'word'))

dtmEs <- DocumentTermMatrix(corpusEsT)
dtmEsTidy <- tidy(dtmEs)
sentimentsEs <- dtmEsTidy %>%
  inner_join(get_sentiments('bing'), by = c(term = 'word'))

# Find most positive reviews
sentiments %>%
  count(document, sentiment, wt = count) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  arrange(-sentiment)

# Find most negative reviews
sentiments %>%
  count(document, sentiment, wt = count) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  arrange(sentiment)

# Most sentiment-contributing words 
sentiments %>%
  count(sentiment, term, wt = count) %>%
  filter(n >= 10) %>%
  mutate(n = ifelse(sentiment == 'negative', -n, n)) %>%
  mutate(term = reorder(term, n)) %>%
  ggplot(aes(term, n, fill = sentiment)) +
  geom_bar(stat = 'identity') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylab('Contribution to sentiment')


