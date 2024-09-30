from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import os
import asyncio
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)
import streamlit as st
from utils import get_query_engine


# Define the events for the workflow
class CompEvent(Event):
    response: str


class ReviewsEvent(Event):
    response: str


class BenefitsEvent(Event):
    response: str


class SurveyEvent(Event):
    response: str


# Define the workflow
class RetentionFlow(Workflow):

    query_engine = get_query_engine()

    @step(pass_context=True)
    async def analyse_comp(self, ctx: Context, ev: StartEvent) -> CompEvent:

        st.session_state['progress_bar'] = st.progress(0, text="Analyzing compensation data...")

        prompt = f"""
        Answer the following questions to the best of your ability and provided data:
        
        1. How does current salary of this employee compare to the industry benchmark? 
        2. Consider starting and current salary of the employee. How does the salary growth compare to industry standard?
        -----------------------------------
        The employee details are as follows:
        {st.session_state['employee_snapshot']}
        """

        response = self.query_engine.query(prompt)

        chunks = []
        # stream the response to the frontend
        # for chunk in st.write_stream(response.response_gen):
        for chunk in response.response_gen:
            chunks.append(chunk)

        # Join all the collected chunks to form the complete response
        full_response = ''.join(chunks)

        # Store the response in the context data
        ctx.data['comp_analysis'] = full_response

        return CompEvent(response=full_response)


    @step(pass_context=True)
    async def analyse_reviews(self, ctx: Context, ev: CompEvent) -> ReviewsEvent:

        st.session_state['progress_bar'].progress(25, text="Analyzing performance reviews...")

        prompt = f"""
                Analyze performance reviews of the employee and provide insights retention recommendations.

                1. Based on the performance reviews, what are the key strengths and areas of improvement for the employee?
                2. How do the performance reviews align with the attrition risk?
                -----------------------------------
                The employee details are as follows:
                {st.session_state['employee_snapshot']}
                """

        response = self.query_engine.query(prompt)

        chunks = []
        # stream the response to the frontend
        # for chunk in st.write_stream(response.response_gen):
        for chunk in response.response_gen:
            chunks.append(chunk)

        # Join all the collected chunks to form the complete response
        full_response = ''.join(chunks)

        # Store the response in the context data
        ctx.data['reviews_analysis'] = full_response

        return ReviewsEvent(response=full_response)


    @step(pass_context=True)
    async def analyse_benefits(self, ctx: Context, ev: ReviewsEvent) -> BenefitsEvent:

        st.session_state['progress_bar'].progress(50, text="Analyzing employee benefits...")

        prompt = f"""
                Analyze the benefits enrollment of the employee and provide insights on retention recommendations.
                
                1. Are there any benefits that the employee is not utilizing? 
                2. Are there any benefits that could be offered to improve employee satisfaction and retention? 
                Use the benefits documentation for reference.
                -----------------------------------
                The employee details are as follows:
                {st.session_state['employee_snapshot']}
                """

        response = self.query_engine.query(prompt)

        chunks = []
        # stream the response to the frontend
        # for chunk in st.write_stream(response.response_gen):
        for chunk in response.response_gen:
            chunks.append(chunk)

        # Join all the collected chunks to form the complete response
        full_response = ''.join(chunks)

        # Store the response in the context data
        ctx.data['benefits_analysis'] = full_response

        return BenefitsEvent(response=full_response)


    @step(pass_context=True)
    async def analyse_survey(self, ctx: Context, ev: BenefitsEvent) -> SurveyEvent:

        st.session_state['progress_bar'].progress(75, text="Analyzing survey results...")

        prompt = f"""
                Analyze the engagement survey responses of the employee and provide insights on retention recommendations.
                
                1. What are the key factors affecting employee engagement and satisfaction?
                2. Are there any areas of improvement based on the survey responses?
                -----------------------------------
                The employee details are as follows:
                {st.session_state['employee_snapshot']}
                """

        response = self.query_engine.query(prompt)

        chunks = []
        # stream the response to the frontend
        # for chunk in st.write_stream(response.response_gen):
        for chunk in response.response_gen:
            chunks.append(chunk)

        # Join all the collected chunks to form the complete response
        full_response = ''.join(chunks)

        # Store the response in the context data
        ctx.data['survey_analysis'] = full_response

        return SurveyEvent(response=full_response)


    @step(pass_context=True)
    async def synthesize_responses(self, ctx: Context, ev: SurveyEvent) -> StopEvent:

        st.session_state['progress_bar'].progress(99, text="Summarizing...")

        prompt = f"""
                Based on the analysis of compensation, performance reviews, benefits enrollment, and engagement survey responses,
                provide a comprehensive retention recommendation for the employee.
                
                For each recommendation you make, provide a rationale about why you're making this recommendation 
                (e.g. employee's compensation is below industry benchmark, or employee raised concerns about long hours in the engagement survey).

                Do not provide recommendation if you are not able to support it with data from analysis.
                
                Keep your recommendations concise and to the point.
                -----------------------------------
                Compensation analysis: {ctx.data['comp_analysis']}
                -----------------------------------
                Performance reviews analysis: {ctx.data['reviews_analysis']}
                -----------------------------------
                Benefits enrollment analysis: {ctx.data['benefits_analysis']}
                -----------------------------------
                Engagement survey analysis: {ctx.data['survey_analysis']}
                
                """

        response = self.query_engine.query(prompt)
        
        # get rid of progress bar
        st.session_state['progress_bar'].empty()

        chunks = []
        # stream the response to the frontend
        for chunk in st.write_stream(response.response_gen):
            chunks.append(chunk)

        # Join all the collected chunks to form the complete response
        full_response = ''.join(chunks)

        return StopEvent(result=full_response)


async def run_workflow():
    w = RetentionFlow(timeout=120, verbose=True)
    result = await w.run()
    return result
