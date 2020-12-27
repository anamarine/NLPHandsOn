print("_______________________________________________________________________")
print("                            NLP deliverable                            ")
print("              Madrid's airbnb reviews, sentiment analysis              ")
print("                        Author: Ana Marin Estañ                        ")
print("_______________________________________________________________________")
print("__ 1. Import data and libraries _______________________________________")

# Check if packages are installed
packages <- c("rstudioapi", "DT", "tm","ggplot2", "ggmap", "wordcloud","RWeka", "reshape2", "tidyverse", "tidytext", "stringr")

install.packages(setdiff(packages, rownames(installed.packages())))  
library(rstudioapi)
library(DT)
library(tm)
library(ggplot2)
library(ggmap)
library(wordcloud)
library(RWeka)
library(reshape2)
library(tidyverse)
library(tidytext)
library(stringr)

#Set work dir
current_path <- getActiveDocumentContext()$path 
setwd(dirname(current_path))

wd <- getwd()
cat("Working directory: ", wd)

#Import data
reviews <- read_csv("reviews.csv")
#Show data head in plot 
datatable(head(reviews, 5))
