# This Dockerfile is for SMT Comp 2023
FROM satcomp-infrastructure:leader
USER root
RUN apt-get update -y --fix-missing \
 && apt-get install -y python3 python3-pip

WORKDIR /

COPY requirements.txt requirements.txt
RUN pip3 install python-sat==0.1.8.dev1 z3-solver==4.12.0
COPY scripts/portfolio/two_stage_portfolio.py /competition/solver
RUN chmod 777 /competition/solver
USER ecs-user