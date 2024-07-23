FROM ubuntu:latest
LABEL authors="wagnertamas"

ENTRYPOINT ["top", "-b"]