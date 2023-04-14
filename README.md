# ELG5164-Group9-MLOPs

This repository contains the scripts utilised for the project "Evaluation of Cloud Cognitive Platforms" in the fullfillment of the requirements of the uOttawa course 
ELG 5164.

The repository is organised in the following structure

  - AWS: Contains CT and CD Notebooks, Dockerfiles and other utility scripts used in SageMaker
  - GCP: Contains CT and CD Notebooks, Dockerfiles and other utility scripts used in Vertex AI
  - Azure: Contains CT and CD Notebooks, Dockerfiles and other utility scripts used in Azure ML
  - SonarQube: Contains utility scripts for setting up SonarQube
  - Endpoint Test: Contains the notebook for testing model hosted on AWS, Azure and GCP

Note: 
1. AWS scripts will not work as the keys have been rotated for security reasons.
2. On May 5th Azure and GCP scripts will cease to work on account of resource clean up in the respective platforms.

The container images can be found at my Docker Hub: https://hub.docker.com/repository/docker/anish26/elg5164-deliverables/general
