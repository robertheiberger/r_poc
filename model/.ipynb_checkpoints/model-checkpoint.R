#library(twitteR)


library(ROAuth)
library(tidyverse)
library(purrrlyr)
library(text2vec)
library(caret)
library(glmnet)
library(ggrepel)
library(dummies)
#library(reticulate)

#sagemaker <- import('sagemaker')
#session <- sagemaker$Session()
#bucket <- session$default_bucket()

prefix <- '/opt/ml'
input_path <- paste(prefix, 'input/data', sep='/')
output_path <- paste(prefix, 'output', sep='/')
model_path <- paste(prefix, 'model', sep='/')
param_path <- paste(prefix, 'input/config/hyperparameters.json', sep='/')

# Channel holding training data
channel_name = 'train'
training_path <- paste(input_path, channel_name, sep='/')

channel_name = 'test'
test_path <- paste(input_path, channel_name, sep='/')

conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")

train <- function() {
    
    #training_files = list.files(path=training_path, full.names=TRUE)
    #train_data = do.call(rbind, lapply(training_files, read.csv))
    train_data = read.csv("/opt/ml/input/data/train/tweets.csv")
    
    #testing_files = list.files(path=training_path, full.names=TRUE)
    #test_data = do.call(rbind, lapply(testing_files, read.csv))
    test_data = read.csv("/opt/ml/input/data/test/tweets.csv")
    
    x_train <- as.character(train_data$tweet)
    x_test <- as.character(test_data$tweet)
    y_train <- train_data$sentiment
    y_test <- test_data$sentiment

    prep_fun <- tolower
    tok_fun <- word_tokenizer

    it_train <- itoken(x_train, preprocessor = prep_fun, tokenizer = tok_fun, progressbar = TRUE)
    it_test <- itoken(x_test, preprocessor = prep_fun, tokenizer = tok_fun, progressbar = TRUE)
    
    vocab <- create_vocabulary(it_train)
    vectorizer <- vocab_vectorizer(vocab)
    dtm_train <- create_dtm(it_train, vectorizer)

    # define tf-idf model
    tfidf <- TfIdf$new()
    
    # fit the model to the train data and transform it with the fitted model
    dtm_train_tfidf <- fit_transform(dtm_train, tfidf)

    # apply pre-trained tf-idf transformation to test data
    dtm_test_tfidf  <- create_dtm(it_test, vectorizer) %>% 
            transform(tfidf)
    
    factor_levels <- lapply(y_train, function(x) {levels(x)})
    
    glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf,
         y = data.matrix(y_train), 
         family = 'multinomial', 
         # L1 penalty
         alpha = 1,
         # interested in the area under ROC curve
         type.measure = "auc",
         # 5-fold cross-validation
         nfolds = 5,
         # high value is less accurate, but has faster training
         thresh = 1e-3,
         # again lower number of iterations for faster training
         maxit = 1e3)
    
    preds <- predict(glmnet_classifier, dtm_test_tfidf, type = 'class')
    
    save(glmnet_classifier, factor_levels, file=paste(model_path, 'glmnet_classifier.RData', sep='/'))
    
    print(table(y_test, preds))
}

# Setup scoring function
serve <- function() {
    app <- plumb(paste(prefix, 'plumber.R', sep='/'))
    app$run(host='0.0.0.0', port=8080)}

# Run at start-up
args <- commandArgs()
if (any(grepl('train', args))) {
    train()}
if (any(grepl('serve', args))) {
    serve()}
