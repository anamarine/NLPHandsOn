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
current_path <- getActiveDocumentContext()$path 
setwd(dirname(current_path))
wdir <- getwd()
cat('Working directory:', wdir)


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

# Keep only English reviews
rowsToKeep = c(reviewsPrep$language == 'english')

reviewsPrep = reviewsPrep[rowsToKeep,]

# Remove empty reviews
reviewsPrep=reviewsPrep[!(is.na(reviewsPrep$comments) | reviewsPrep$comments==""), ]


#__ 4. Load and inspect Corpus _________________________________________
corpus = Corpus(VectorSource(reviewsPrep$comments))
cat('Corpus length:', length(corpus), 'entries')
if(length(corpus) > 1000)
  corpus=corpus[1:1000]
  cat('Limit corpus length to:', length(corpus), 'entries')
summary(corpus[1:5])
inspect(corpus[[1]])
meta(corpus[[1]])
content(corpus[[1]])


#__ 5. Create and transform DTM ________________________________________

myStopwords = c(stopwords(),"airbnb","very", "madrid","stay")
corpusTransf <- tm_map(corpus, tolower)
corpusTransf <- tm_map(corpusTransf, removeNumbers)
corpusTransf <- tm_map(corpusTransf, removeWords, myStopwords)
corpusTransf <- tm_map(corpusTransf, stemDocument)
corpusTransf <- tm_map(corpusTransf, removePunctuation)
corpusTransf <- tm_map(corpusTransf, stripWhitespace)

# Compare transdormed documents
corpus[["100"]][["content"]]; corpusTransf[["100"]][["content"]]

#__ 6. Create TDM ______________________________________________________
# create TDM with TF-IDF weighting
tdm = TermDocumentMatrix(corpusTransf, control = list(weighting = weightTfIdf))
tdm
#__ 7. Basic text analysis _____________________________________________

#TF-IDF word frequencies
freq=rowSums(as.matrix(tdm))
plot(sort(freq, decreasing = TRUE),col='red',main='Word TF-IDF frequencies', xlab='TF-IDF-based rank', ylab = 'TF-IDF')
tail(sort(freq),n=10)

#Word Cloud
pal=brewer.pal(8,"Reds")
unigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
tdm.unigram = TermDocumentMatrix(corpusTransf,
                                control = list (weighting = weightTfIdf,
                                                tokenize = unigramTokenizer))
freq = sort(rowSums(as.matrix(tdm.unigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = FALSE, colors=pal)


#__ 8. Sentiment analysis ______________________________________________
dtm <- DocumentTermMatrix(corpusTransf)

dtmTidy <- tidy(dtm)
sentiments <- dtmTidy %>%
  inner_join(get_sentiments("bing"), by = c(term = "word"))

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

# Most contributive words to sentiments
sentiments %>%
  count(sentiment, term, wt = count) %>%
  filter(n >= 10) %>%
  mutate(n = ifelse(sentiment == "negative", -n, n)) %>%
  mutate(term = reorder(term, n)) %>%
  ggplot(aes(term, n, fill = sentiment)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylab("Contribution to sentiment")


