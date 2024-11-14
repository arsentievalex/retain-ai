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
    """Event class to capture compensation analysis results."""
    response: str


class ReviewsEvent(Event):
    """Event class to capture performance reviews analysis results."""
    response: str


class BenefitsEvent(Event):
    """Event class to capture benefits analysis results."""
    response: str


class SurveyEvent(Event):
    """Event class to capture survey analysis results."""
    response: str


# Define the workflow for employee retention analysis
class RetentionFlow(Workflow):
    """
    Workflow class to analyze various aspects (compensation, performance reviews, benefits, surveys) of an employee's data 
    and provide retention recommendations based on aggregated analysis.
    """
    
    # Initialize query engine for LLM-based analysis
    query_engine = get_query_engine()

    @step(pass_context=True)
    async def analyse_comp(self, ctx: Context, ev: StartEvent) -> CompEvent:
        """
        Analyzes the compensation data for the employee and returns a compensation analysis event.

        Parameters:
            ctx (Context): Workflow context to store intermediate data.
            ev (StartEvent): Starting event to trigger the compensation analysis.

        Returns:
            CompEvent: Contains the analysis response.
        """
        # Display progress for the compensation analysis
        st.session_state['progress_bar'] = st.progress(0, text="Analyzing compensation data...")

        # Define prompt with questions on salary comparison and growth for the employee
        prompt = f"""
        Answer the following questions to the best of your ability and provided data:
        
        1. How does current salary of this employee compare to the industry benchmark? 
        2. Consider starting and current salary of the employee. How does the salary growth compare to industry standard?
        -----------------------------------
        The employee with high risk of attrition is:
        {st.session_state['employee_snapshot']}
        """

        # Query the language model with the prompt
        response = self.query_engine.query(prompt)

        # Stream and collect the response chunks for real-time frontend updates
        chunks = []
        for chunk in response.response_gen:
            chunks.append(chunk)

        # Join chunks to form a complete response string
        full_response = ''.join(chunks)

        # Store the full response in context for downstream use
        ctx.data['comp_analysis'] = full_response

        return CompEvent(response=full_response)

    @step(pass_context=True)
    async def analyse_reviews(self, ctx: Context, ev: CompEvent) -> ReviewsEvent:
        """
        Analyzes the performance reviews for the employee and returns a reviews analysis event.

        Parameters:
            ctx (Context): Workflow context to store intermediate data.
            ev (CompEvent): Event containing compensation analysis results.

        Returns:
            ReviewsEvent: Contains the analysis response.
        """
        # Update progress bar to show analysis status
        st.session_state['progress_bar'].progress(25, text="Analyzing performance reviews...")

        # Define prompt to analyze performance reviews and provide retention insights
        prompt = f"""
                Analyze performance reviews of the employee and provide insights retention recommendations.

                1. Based on the performance reviews, what are the key strengths and areas of improvement for the employee?
                2. How do the performance reviews align with the attrition risk?
                -----------------------------------
                The employee with high risk of attrition is:
                {st.session_state['employee_snapshot']}
                """

        # Query the language model with the prompt
        response = self.query_engine.query(prompt)

        # Stream and collect response chunks
        chunks = []
        for chunk in response.response_gen:
            chunks.append(chunk)

        # Form the complete response and store in context
        full_response = ''.join(chunks)
        ctx.data['reviews_analysis'] = full_response

        return ReviewsEvent(response=full_response)

    @step(pass_context=True)
    async def analyse_benefits(self, ctx: Context, ev: ReviewsEvent) -> BenefitsEvent:
        """
        Analyzes the benefits enrollment for the employee and returns a benefits analysis event.

        Parameters:
            ctx (Context): Workflow context to store intermediate data.
            ev (ReviewsEvent): Event containing performance reviews analysis results.

        Returns:
            BenefitsEvent: Contains the analysis response.
        """
        # Update progress bar for benefits analysis
        st.session_state['progress_bar'].progress(50, text="Analyzing employee benefits...")

        # Define prompt to assess benefits usage and potential improvements for retention
        prompt = f"""
                Analyze the benefits enrollment of the employee and provide insights on retention recommendations.
                
                1. Are there any benefits that the employee is not utilizing? 
                2. Are there any benefits that could be offered to improve employee satisfaction and retention? 
                Use the benefits documentation for reference.
                -----------------------------------
                The employee with high risk of attrition is:
                {st.session_state['employee_snapshot']}
                """

        # Query the language model with the prompt
        response = self.query_engine.query(prompt)

        # Stream and collect response chunks
        chunks = []
        for chunk in response.response_gen:
            chunks.append(chunk)

        # Form the complete response and store in context
        full_response = ''.join(chunks)
        ctx.data['benefits_analysis'] = full_response

        return BenefitsEvent(response=full_response)

    @step(pass_context=True)
    async def analyse_survey(self, ctx: Context, ev: BenefitsEvent) -> SurveyEvent:
        """
        Analyzes the engagement survey results for the employee and returns a survey analysis event.

        Parameters:
            ctx (Context): Workflow context to store intermediate data.
            ev (BenefitsEvent): Event containing benefits analysis results.

        Returns:
            SurveyEvent: Contains the analysis response.
        """
        # Update progress bar for survey analysis
        st.session_state['progress_bar'].progress(75, text="Analyzing survey results...")

        # Define prompt to interpret survey data and suggest retention improvements
        prompt = f"""
                Analyze the engagement survey responses of the employee and provide insights on retention recommendations.
                
                1. What are the key factors affecting employee engagement and satisfaction?
                2. Are there any areas of improvement based on the survey responses?
                -----------------------------------
                The employee with high risk of attrition is:
                {st.session_state['employee_snapshot']}
                """

        # Query the language model with the prompt
        response = self.query_engine.query(prompt)

        # Stream and collect response chunks
        chunks = []
        for chunk in response.response_gen:
            chunks.append(chunk)

        # Form the complete response and store in context
        full_response = ''.join(chunks)
        ctx.data['survey_analysis'] = full_response

        return SurveyEvent(response=full_response)

    @step(pass_context=True)
    async def synthesize_responses(self, ctx: Context, ev: SurveyEvent) -> StopEvent:
        """
        Synthesizes the analysis from compensation, reviews, benefits, and survey data 
        to generate a final retention recommendation for the employee.

        Parameters:
            ctx (Context): Workflow context to store intermediate data.
            ev (SurveyEvent): Event containing survey analysis results.

        Returns:
            StopEvent: Contains the final retention recommendations.
        """
        # Update progress bar for final synthesis step
        st.session_state['progress_bar'].progress(99, text="Summarizing...")

        # Define prompt for synthesizing comprehensive retention recommendations
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

        # Query the language model with the final prompt
        response = self.query_engine.query(prompt)

        # Clear progress bar after final analysis
        st.session_state['progress_bar'].empty()

        # Stream and collect response chunks for final output
        chunks = []
        for chunk in st.write_stream(response.response_gen):
            chunks.append(chunk)

        # Form the complete recommendation response
        full_response = ''.join(chunks)

        return StopEvent(result=full_response)


async def run_workflow():
    """
    Initiates and runs the RetentionFlow workflow with a specified timeout and verbosity.
    
    Returns:
        StopEvent: Final event containing comprehensive retention recommendations.
    """
    w = RetentionFlow(timeout=120, verbose=True)
    result = await w.run()
    return result

