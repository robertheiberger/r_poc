FROM ubuntu:16.04

MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libssl-dev \
    libopenblas-dev \
    liblapack-dev \
    libopencv-dev \
    libxt-dev \
    libpython3-dev \
    python3-venv \
    python3 \
    ca-certificates
    
RUN R -e "install.packages(c('mda', 'plumber','twitteR','ROAuth','tidyverse','purrrlyr','text2vec','caret','glmnet','ggrepel','dummies','reticulate'), repos='https://cloud.r-project.org')"

RUN Rscript -e "install.packages(c('mda', 'plumber'),repos='https://cloud.r-project.org')"
RUN Rscript -e "install.packages(c('tidyverse'),repos='https://cloud.r-project.org')"

COPY model.R /opt/ml/model.R
COPY plumber.R /opt/ml/plumber.R

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/model.R", "--no-save"]
