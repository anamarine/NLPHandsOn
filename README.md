# NLP deliverable. Intelligent Systems, 2020-21

**Author:** Ana Marin Estañ

**Master course:** M.Sc., Health and Medical Data Analytics (HMDA)

# Instructions to run the deliverable

Two options are provided to check and run the project. Both options contain the same code.
It is important that the application file is run in the same directory that the data folder, which contains the ```reviews.csv``` file. 

## Option 1. R script
The R script,  ```NLPDeliverable.R```, can be easily executed in an IDE like RStudio.

## Option 2. R notebook
A notebook (``NLP.Rmd``) and an html file (``NLP.nb.html``) are additionally provided to make the understanding of the application easier. Moreover, the cells are executed so the results are already provided. 

## Data sampling
Both options are preset by default to only get a 1% random sample of the data This way the execution of the code is fast. 
However, if it is desired to run the application with all the data, it can be easily done by changing the variable ``sampleReviews`` to ``FALSE``. Which declaration corresponds to the first line of code (line 11 and 16 in the R script and the notebook, respectively).
