FROM python:2.7.16-stretch
COPY . /quasinet
RUN apt update
RUN apt-get -y install graphviz libgraphviz-dev pkg-config
RUN apt -y install r-base-core
RUN pip install quasinet
RUN Rscript /quasinet/download_R_packages.R