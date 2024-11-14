# RetainAI


**RetainAI** is a web-based application designed to help managers predict employee attrition and generate personalized retention strategies. Utilizing machine learning (ML) and advanced large language models (LLMs), RetainAI provides actionable, data-driven insights, empowering businesses to address turnover proactively and enhance employee satisfaction through customized retention recommendations.

[Watch the demo video on YouTube](https://youtu.be/yS0RuteEBoI)


## Overview
Employee turnover remains a significant cost factor, with U.S. businesses incurring approximately $1 trillion in losses annually due to employee attrition. Many managers are responsible for large teams, making it challenging to monitor employee sentiment and develop tailored retention strategies. RetainAI tackles these challenges by offering an intelligent, streamlined approach to predict attrition risk and create personalized retention strategies, helping companies retain top talent.


## Key Features
- **Data Uploads**: Allows users to upload CSV files with employee data or relevant PDF documents (e.g., benefits documentation, industry trends) to enrich the AI's understanding and generate more personalized recommendations.

- **Field Mapping UI** This interface allows users to map their custom CSV columns to the required fields used by the attrition prediction model. The app includes a semi-automated mapping process powered by fuzzy matching, which can automatically align similar column names (e.g., mapping "Employee Name" to "Full Name"). Users can review and adjust mappings manually if needed to ensure compatibility with the model.
  
- **Attrition Prediction**: Predicts employee attrition risk using a Random Forest Classifier based on factors such as tenure, compensation, promotion history, and performance metrics.

- **Dashboard Overview**: A user-friendly dashboard, provides managers with a summary of attrition risks across their teams, highlighting key risk factors and enabling easy access to detailed retention strategies.

- **Personalized Retention Strategy with Generative AI**: Provides detailed retention strategies by analyzing various aspects of employee data, including performance reviews, compensation history, benefits utilization, and engagement surveys. Generative AI enhances these insights by contextualizing unstructured information, enabling more nuanced and targeted recommendations.

- **PDF Export**: Exports detailed retention strategies and summaries as PDF files, allowing for documentation and easy sharing within the organization.

- **Interactive Chat Mode**: Enables managers to interact directly with the AI, asking questions about specific employees or discussing retention strategies. This mode leverages a conversational workflow, allowing for in-depth analysis and real-time feedback.


## Why Use Gen AI?
Traditional attrition prediction techniques are highly effective for analyzing structured data and predicting attrition risk based on factors like tenure, compensation, and performance metrics. However, these models often fall short when it comes to incorporating unstructured, qualitative data, which can provide deeper insights into the reasons behind employee attrition.

RetainAI combines these traditional predictive methods with the power of generative AI. While the Random Forest Classifier handles the structured data to assess attrition probability, the generative AI component extracts valuable insights from unstructured qualitative data. This includes employee engagement comments, performance reviews, and industry trend PDFs, allowing the app to create highly personalized and context-rich retention strategies. By synthesizing both structured and unstructured data, RetainAI delivers more nuanced and actionable recommendations, helping managers address attrition risks with a comprehensive, data-driven approach.

## Architecture
![Image description](/img/retainai_architecture.png)

- **Frontend**: Built with Streamlit, offering an accessible, web-based interface that allows managers to view, interact with, and manage team data and attrition insights.

- **Backend**: Hosted on NVIDIA AI Workbench, featuring a Random Forest Classifier for attrition prediction, and a Llama-Index workflow with llama-3.1-70b-instruct from the NVIDIA API Catalog to generate retention strategies and chat functionality. 

- **Data Handling**: RetainAI allows flexibility for users to either work with sample data or upload their own data files, enabling a customized and relevant analysis experience.

*CSV Uploads*: Users can upload their own CSV files for attrition prediction, as long as they match the required column names and formats. This ensures accurate model predictions based on data specific to their organization.

*PDF Uploads*: Users can also upload PDF files containing qualitative data, such as industry reports or benefits docs specific to their company. These documents are processed using RAG engine, which stores embedded representations in a local vector database. This enables the LLM to retrieve and analyze relevant information from these unstructured sources to enrich retention strategies.

*Sample Data*: For users looking to explore the app's features without uploading their own files, RetainAI provides sample CSV and PDF data. This includes fictional employee records and sample documents, allowing users to quickly get started and experience the app's full functionality.


## Workflow
Below is representation of the Llama-Index workflow used for retention recommendation:
![Image description](/img/retainai_workflow.png)


## Getting Started
1. Download and install [NVIDIA AI Workbench](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/)

2. Obtain an API key from [NVIDIA API Catalog](https://build.nvidia.com/explore/discover)

3. Open NVIDIA AI Workbench on your machine, select "Clone Project" and paste the below repository URL.

``https://github.com/arsentievalex/retain-ai.git``

4. Set up your NVIDIA API KEY in secrets.

5. Start Environment and click "Open Streamlit" to run the web app.

6. Upload your CSV and PDF files on the "Data Upload" page, or use the sample data provided. The uploaded CSV files should contain the required columns and data types. For more information, refer to the FAQ page in the web app.

7. Once the data is loaded, navigate to the "Dashboard" to view the list of employees with their predicted attrition probabilities. Employees with an attrition probability greater than 50% are marked as high risk (indicated with an exclamation point).

8. In the "AP Methodology" tab, you can view the key factors the model uses to predict employee attrition. Each factor’s importance score shows how much it influences the likelihood of an employee leaving.

9. Select an employee and click "Retention Recommendation." The LLM model will generate a personalized retention strategy, which you can download as a PDF file.

10. On the "Chat" page, you can ask questions about your entire team (not just individual employees). For example, you might ask, "Which employees are not satisfied with work-life balance?" or "Which employees performed poorly in the Q3 performance review?"
