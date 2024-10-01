# RetainAI

**RetainAI** is a web-based application designed to help managers predict employee attrition and generate personalized retention strategies. By using machine learning and large language models (LLMs), RetainAI equips managers with data-driven insights into employee retention, helping businesses reduce turnover costs and improve employee satisfaction.

[Watch the demo video on YouTube](https://youtu.be/yS0RuteEBoI)

## Description
Employee turnover is a major financial challenge for businesses, costing U.S. companies approximately $1 trillion annually. Managers often oversee large teams, making it difficult to track employee sentiment and implement retention strategies. RetainAI was built to address these challenges by providing a streamlined way to predict attrition and offer personalized strategies for retaining valuable employees.

## Features
- **Attrition Prediction**: Uses a Random Forest Classifier model to predict the probability of employee attrition based on key features such as tenure, compensation, promotion history, and performance.
- **Personalized Retention Strategy**: Generates a comprehensive, personalized retention strategy for employees at risk of attrition, leveraging data such as performance reviews, compensation history, benefits usage, and engagement surveys.
- **Dashboard**: A user-friendly dashboard built with Streamlit, providing managers with an overview of their team's attrition risk.
- **PDF Export**: Allows managers to export detailed retention strategies as PDF files for future reference.

## Architecture

<img src="https://i.postimg.cc/mkKDNK7k/retainai-architecture.png"/>

- **Frontend**: Built with Streamlit, allowing users to interact with the app via a simple, web-based interface.
- **Backend**: Hosted on NVIDIA AI Workbench, using a Random Forest Classifier for attrition prediction and a Llama-Index workflow with llama-3.1-70b-instruct hosted on NVIDIA API catalogue to generate retention strategies.

Below is representation of the Llama-Index workflow:

<img src="https://i.postimg.cc/GmF2G7t3/retainai-workflow.png"/>

- **Data**: Includes employee details such as tenure, compensation, performance scores, and promotion history. For the demo, a fictional employee dataset is used. Additionaly, 2 PDF files with unstructured data are embedded, stored in local vector db and used for RAG by the model.

## Getting Started
1. Download and install [NVIDIA AI Workbench](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/)

2. Clone the repository

``git clone https://github.com/arsentievalex/retain-ai.git``

3. Set up your NVIDIA API KEY in secrets.


