FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install basic dependencies
RUN apt update && apt install -y python3-pip git vim sudo curl wget apt-transport-https ca-certificates gnupg libgl1 cmake build-essential patchelf

# Install Eigen 3.4+ for mujoco build (Ubuntu 22.04 has 3.4.0)
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar -xzf eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && mkdir build && cd build && \
    cmake .. && make install && \
    cd ../.. && rm -rf eigen-3.4.0 eigen-3.4.0.tar.gz

# Install Miniconda (specific version with Python 3.10)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

RUN conda config --add channels anaconda && \
    conda config --add channels conda-forge && \
    conda config --add channels nvidia && \
    conda config --add channels r && \
    conda config --add channels bioconda && \
    conda config --add channels defaults
    # conda config --set channel_priority strict

# probably delete
# RUN conda config --set solver classic

# Create and activate conda environment (explicitly pin Python version)
RUN conda create -n myenv python=3.10.14 -c conda-forge --override-channels -y && \
    conda clean -a

# Install MuJoCo library 2.3.7 (required for mujoco==2.3.7)
RUN mkdir -p /opt/mujoco && \
    wget https://github.com/deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz && \
    tar -xzf /tmp/mujoco.tar.gz -C /opt/mujoco --strip-components=1 && \
    rm /tmp/mujoco.tar.gz

# Set MUJOCO_PATH environment variable
ENV MUJOCO_PATH=/opt/mujoco
ENV MUJOCO_PLUGIN_PATH=${MUJOCO_PATH}/plugin
ENV LD_LIBRARY_PATH=${MUJOCO_PATH}/bin:${LD_LIBRARY_PATH}

# Install all dependencies in the conda environment
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate myenv && \
    pip install setuptools && \
    pip uninstall -y jax jaxlib jaxopt flax brax && \
    pip install --upgrade 'jax[cuda]==0.4.16' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install --upgrade jaxlib==0.4.16 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install --upgrade flax==0.6.11 optax==0.1.7 jaxopt==0.8.1 brax==0.9.2 chex==0.1.8 && \
    pip install notebook matplotlib tqdm jupyter ipython wandb rich && \
    pip install mujoco==3.1.6 --only-binary :all: && \
    pip install distrax==0.1.5 gym==0.26.2 gymnax==0.0.6 && \
    pip install tensorflow-probability==0.22.0 scipy==1.11.3 && \
    pip install --upgrade typing_extensions && \
    pip install --upgrade wandb pydantic"

# Copy requirements files to the container
COPY requirements_conda.txt /tmp/requirements_conda.txt
COPY requirements_pip.txt /tmp/requirements_pip.txt

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate myenv && conda install --file /tmp/requirements_conda.txt"
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate myenv && pip install -r /tmp/requirements_pip.txt"

# Install or upgrade jax and jaxlib after requirements installation
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate myenv && \
    pip install --upgrade jax==0.4.26 && \
    pip install --upgrade 'jaxlib==0.4.26+cuda12.cudnn89' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"


# --------------------------------------------------------------------------#
# # Install utilities as root user
# RUN apt-get update && apt-get install -y p7zip-full unrar htop

# # Create and configure non-root user
# # ARG UID=1000
# ARG UID=3745
# RUN useradd -u $UID --create-home duser && \
#     echo "duser:duser" | chpasswd && \
#     adduser duser sudo && \
#     mkdir -p /home/duser/.local/bin && \
#     chown -R duser:duser /home/duser

# # Create /app directory and give access to duser
# # RUN mkdir -p /app && chown -R duser:duser /app
# RUN mkdir -p /app && chown -R duser:duser /app && chmod -R 777 /app

# # Switch to non-root user
# USER duser
# WORKDIR /home/duser/

# # Set correct permissions on working directory
# RUN chown -R duser:duser /app

# --------------------------------------------------------------------------#

# Set user UID to match host UID
ARG UID=3745

# Create user with specified UID
RUN useradd -u $UID --create-home duser && \
    echo "duser:duser" | chpasswd && \
    adduser duser sudo && \
    mkdir -p /home/duser/.local/bin && \
    chown -R duser:duser /home/duser

USER duser
# WORKDIR /home/duser/

WORKDIR /app

# --------------------------------------------------------------------------#

# Configure git
RUN git config --global user.email "george_nigm@icloud.com"
RUN git config --global user.name "george_nigm"

# Add alias for ipython
RUN echo "alias i='/usr/local/bin/ipython'" >> ~/.bashrc

# Set default shell for conda environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Set PATH for local binaries
RUN echo 'export PATH=$PATH:/home/duser/.local/bin' >> ~/.bashrc

# ENTRYPOINT ["conda", "run", "-n", "myenv", "python"]





# Commands
#1 docker rmi georgenigm_docker
#2.1. docker build -t georgenigm_docker .
#2.2. docker build -f georgenigm_docker --build-arg UID=$UID --build-arg GIT_TOKEN=$GIT_TOKEN -t georgenigm_image .




#3.1. docker run --gpus '"device=1"' -d -it -v $(pwd):/app --name georgenigm_docker_container georgenigm_docker /bin/bash





#3.2. docker run --rm --gpus '"device=2,3,5"' -v $(pwd):/app --name ${USER}_tutorialcontainer ${USER}_dockertutorial


#3.3. docker run --rm -d --gpus '"device=2,3,5"' -v $(pwd):/app --name ${USER}_tutorialcontainer ${USER}_dockertutorial /bin/bash -c "python -u script.py &> log.txt"




# On my personal computer – I connect via ssh to a shared server (entering passwords) – on the server I launch the Docker container – from the container I start jupyter notebook


# docker run --rm --gpus '"device=0,1,2"' -d -it -v $(pwd):/app --name ${USER}_container ${USER}_docker /bin/bash





# docker run --rm --gpus '"device=0,1,2"' -d -v $(pwd):/app --name georgenigm_docker_job georgenigm_docker /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario.py &> job.log"



# conda activate myenv




# docker run --gpus '"device=0"' -d -it -v $(pwd):/app --name georgenigm_docker_container georgenigm_docker /bin/bash

# docker run --rm --gpus '"device=0,1,2,3,4"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario.py &> job.log"

# docker run --rm --gpus all -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario.py &> job.log"








# Commands 30-jun-25

#1. docker rmi georgenigm_docker

#2. docker build -t georgenigm_docker .

#3.1. docker run --gpus '"device=7"' -d -it -v $(pwd):/app --name georgenigm_viz georgenigm_docker /bin/bash

# docker run --gpus '"device=7"' -d -it -v $(pwd):/app --name georgenigm_viz georgeoni40/georgenigm_docker:latest /bin/bash




#3.2. docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_buy_0_1_2_3_4_5_6_7 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl.py &> job.log"

#3.2. docker run --rm --gpus '"device=4,5,6,7"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_buy_1 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl.py &> job.log"

#3.3. docker run --rm --gpus '"device=6,7"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u historical_scenario.py &> job.log"


# docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_aggressive_buy_0_1_2_3_4_5_6_7 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl_copy.py"


# docker run --rm --gpus '"device=4,5,6,7"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_aggressive_sell_75_4_5_6_7 georgeoni40/georgenigm_docker:latest conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl_copy.py --config 1_run_exp_aggresive_scenario_s5_sell"





# with config! 

# docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_aggressive_buy_0_1_2_3_4_5_6_7 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl_copy.py --config 1_run_exp_aggresive_scenario"






# docker run --rm --gpus '"device=1"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_historical_buy_300_1 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u heuristic_historical_scenario_run.py"

# docker run --rm --gpus '"device=2"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_historical_buy_485_2 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u heuristic_historical_scenario_run.py"

# docker run --rm --gpus '"device=3"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_heuristic_buy_1400_3 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u heuristic_historical_scenario_run.py"



# docker run --rm --gpus '"device=4"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_heuristic_sell_300_4 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u heuristic_historical_scenario_run.py"

# docker run --rm --gpus '"device=5"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_heuristic_sell_485_5 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u heuristic_historical_scenario_run.py"

# docker run --rm --gpus '"device=7"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_heuristic_sell_1400_7 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u heuristic_historical_scenario_run.py"


# 300, 485, 1400 - in run
# 695, 865 - not in run






# with config! 

# docker run --rm --gpus '"device=6"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_aggressive_buy_75_6_autoreg georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u /homes/80/georgenigm/LOBS5/1_run_exp_aggressive_scenario_whole_lvl_autoreg_26_01_12.py --config 1_run_exp_aggresive_scenario" 



# && - to glue commands together

# docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_aggressive_buy_75_0_1_2_3_4_5_6_7 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl_copy.py --config 1_run_exp_aggresive_scenario_buy_75" && docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_aggressive_sell_75_0_1_2_3_4_5_6_7 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl_copy.py --config 1_run_exp_aggresive_scenario_sell_75"




# docker run --rm --gpus '"device=6"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_step_4_sample_day_mapping georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u step_4_sample_day_mapping.py &> job.log"



# docker run --rm --gpus '"device=0,1"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name historical_scenario_run_quantile_fixing_buy_485_2 georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u heuristic_historical_scenario_run_quantile_fixing.py"





# docker run --rm --gpus '"device=0,1"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_aggressive_buy_75_0_1_autoreg georgeoni40/georgenigm_docker:latest conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl_autoreg_26_01_14.py --config 1_run_exp_aggresive_scenario_autoreg"


# docker run --rm --gpus '"device=0,1"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f --name georgenigm_exp_aggressive_buy_75_0_1_autoreg georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl_autoreg_26_01_14.py --config 1_run_exp_aggresive_scenario_autoreg"





# Terminal 1: GPU 0
# docker run --rm --gpus '"device=0"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl_autoreg_26_01_14.py --config 1_run_exp_aggresive_scenario_autoreg_bs8_gpu0"

# Terminal 2: GPU 1
# docker run --rm --gpus '"device=1"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl_autoreg_26_01_14.py --config 1_run_exp_aggresive_scenario_autoreg_bs8_gpu1"

# Terminal 3: GPU 2
# docker run --rm --gpus '"device=2"' -v $(pwd):/app -e WANDB_API_KEY=74075d19681454163130e79756ce47db4dcb571f georgenigm_docker conda run -n myenv /bin/bash -c "cd /app && python -u 1_run_exp_aggressive_scenario_whole_lvl_autoreg_26_01_14.py --config 1_run_exp_aggresive_scenario_autoreg_bs8_gpu2"