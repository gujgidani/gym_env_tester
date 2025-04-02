FROM pytorch/pytorch

# Install sumo
RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    sumo-doc \
    git

RUN export WANDB_API_KEY=0e1cbafc7bcb99295fb33871250924701333b1c2

# Install python packages
RUN pip install \
    numpy \
    wandb \
    traci \
    gymnasium[all] \
    onnx \
    onnxruntime

RUN wandb login

RUN git clone https://github.com/wagnertamas/gym_env_tester.git

WORKDIR /gym_env_tester

CMD ["wandb agent wagnertamas-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/traffic-light-dqn/d3nvb3he"]