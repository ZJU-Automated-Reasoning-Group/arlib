FROM ubuntu:20.04

RUN apt-get update \

 && apt-get install -y python3 python3-pip \

 && pip3 install --upgrade pip \

 && pip3 install pyinstaller


ENTRYPOINT ["/bin/bash", "setup.sh"]

RUN pyinstaller -F scripts/portfolio/two_stage_portfolio.py \
 && cp dist/two_stage_portfolio solver
