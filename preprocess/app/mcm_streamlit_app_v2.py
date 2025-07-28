#!/usr/bin/env python3
r"""
MCM 2025 Conference Streamlit App

This application allows users to input keywords and research interests to generate
a personalized mini-program book with relevant talks and sessions.


Change Chapter 1 to a format like MCM_ProgramBook_TEX/Schedule.tex, but with some differences.  Let me give an example using Mon Morning. it has opening,  plenary talks, coffee breaks, lunch. Include all of theem. But then for parallel sessions marked by S1, S2, S3, S4, and T1, read in the special session abstracts S1.tex, S2.tex, ..., S5.tex, as well as the talks in S1-1.tex, ...,S1-4.tex in session S1; S2-1.tex, ..., S2-4.tex in S2; T1-1.tex,... (note T1.tex does not exists for technical sessions) etc. Then use the AI model to choose the most fitting session out of the five parallel talks.  

Suppose AI model decides that S4 is the best session, then we just need to simplify lines  2 to 109 schedule.tex to "\vspace{-10ex}
\hspace*{-0.5cm}\begin{sideways}\footnotesize\begin{tabularx}{\textheight}{l*{\numcols}{|Y}}
\TableHeading{ \hspace*{-2.5cm}Mon, Jul 28, 2025 -- Morning }
\label{MonMorning}
\\\hline
\TableEvent{08:00--17:30}{Conference Check-In, HH Lobby}\\
\OpeningClosingEvent{08:45--09:00}{Opening Ceremony by Fred Hickernell, Nicole Beebe, and Kevin Corlette, HH Auditorium}\\
\input{sessP1.tex}
\TableEvent{10:00--10:30}{Coffee Break, HH Lobby}\\
\rowcolor{\SessionTitleColor}
&\tableSpecialCL{WH Auditorium}
{Hardware or Software for (Quasi-)Monte Carlo Algorithms, Part I}
{S4}
{Mike Giles}
\\\hline"

Then in Chapter 2, we include all talks abstracts from plenary talks and S4-1.tex,... S4-4.tex in  preprocess/input/abstracts.

Repeat for Mon afternoon, Tue morning, ...., Friday morning in schedule.tex to come up with the personal schedule  in preprocess/app/mcm_streamlit_app_v2.py

The main logic is in ai_select_best_session()

Author: Sou-Cheng Choi, SouLab, LLC
"""
import time
import os
import sys
import csv
import re
import subprocess
import tempfile
import shutil
import traceback
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

# Global debug flag
IS_DEBUG_APP = False
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import json
import time
import requests


# cd /Users/terrya/Documents/ProgramData/MCM-2025-Program/preprocess/app
# streamlit run mcm_streamlit_app.py --server.port 8502

class LLMAssistant:
    """Local LLM integration using Ollama for intelligent talk selection and program generation"""
    
    def __init__(self, model_name="qwen3", fallback_model="gemma3n"):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.base_url = "http://localhost:11434"
        self.available = self.check_ollama_availability()
        self.evaluation_cache = {}  # Cache for talk evaluations
        
        # Batch processing configuration
        self.batch_config = {
            'min_batch_size': 3,
            'max_batch_size': 15,
            'default_small': 5,    # For ≤20 talks
            'default_medium': 8,   # For 21-100 talks  
            'default_large': 12,   # For >100 talks
            'timeout_threshold': 30,  # Seconds before reducing batch size
            'fast_threshold': 5,      # Seconds for considering increase
            'content_length_threshold': 800  # Characters for reducing batch size
        }
    
    def check_ollama_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if IS_DEBUG_APP:
                    print(f"DEBUG: Available models: {model_names[:5]}...")  # Show first 5
                if IS_DEBUG_APP:
                    print(f"DEBUG: Looking for model: {self.model_name}")
                
                # Try exact match first
                if self.model_name in model_names:
                    if IS_DEBUG_APP:
                        print(f"DEBUG: Found exact model match: {self.model_name}")
                    return True
                
                # Try with :latest suffix
                model_with_latest = f"{self.model_name}:latest"
                if model_with_latest in model_names:
                    if IS_DEBUG_APP:
                        print(f"DEBUG: Found model with :latest tag: {model_with_latest}")
                    self.model_name = model_with_latest  # Update to use the full name
                    return True
                
                # Try fallback model
                if self.fallback_model in model_names:
                    if IS_DEBUG_APP:
                        print(f"DEBUG: Using fallback model: {self.fallback_model}")
                    self.model_name = self.fallback_model
                    return True
                
                # Try fallback model with :latest suffix
                fallback_with_latest = f"{self.fallback_model}:latest"
                if fallback_with_latest in model_names:
                    if IS_DEBUG_APP:
                        print(f"DEBUG: Using fallback model with :latest tag: {fallback_with_latest}")
                    self.model_name = fallback_with_latest
                    return True
                
                if IS_DEBUG_APP:
                    print(f"DEBUG: Neither {self.model_name} nor {self.fallback_model} found (with or without :latest)")
                    
            return False
        except Exception as e:
            if IS_DEBUG_APP:
                print(f"DEBUG: Error checking Ollama: {e}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using local LLM"""
        if not self.available:
            return "LLM service unavailable. Please start Ollama and ensure models are installed."
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0 # reproducible
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def analyze_research_interests(self, interests: Dict) -> str:
        """Analyze user's research interests and provide insights"""
        prompt = f"""
                You are an expert conference program advisor for the MCM 2025 (Monte Carlo Methods) conference.

                User's Research Profile:
                - Keywords: {', '.join(interests.get('keywords', []))}
                - Research Areas: {', '.join(interests.get('areas', []))}
                - Experience Level: {interests.get('experience', 'Not specified')}
                - Preferences: {interests.get('preferences', 'None specified')}

                Please provide:
                1. A brief analysis of their research focus
                2. 3-5 specific talk types or session topics they should prioritize
                3. Networking suggestions for this research profile
                4. Any emerging trends in their field they should be aware of

                Keep the response concise and actionable (max 300 words).
                """
        st.debug(f"DEBUG: {prompt = }")
        return self.generate_response(prompt, max_tokens=400)
    
    def suggest_talk_selection_strategy(self, interests: Dict, available_talks_count: int) -> str:
        """Suggest an optimal talk selection strategy"""
        prompt = f"""
                You are helping a researcher at the MCM 2025 conference optimize their schedule.

                Research Profile:
                - Keywords: {', '.join(interests.get('keywords', []))}
                - Areas: {', '.join(interests.get('areas', []))}
                - Experience: {interests.get('experience', 'Not specified')}

                Conference Context:
                - Available relevant talks: {available_talks_count}
                - Conference duration: 5 days
                - Typical schedule: 8-10 sessions per day with parallel tracks

                Provide a strategic recommendation for:
                1. How many talks to select per day (considering fatigue and note-taking)
                2. Balance between core expertise and exploratory topics
                3. Time for networking vs. attending talks
                4. Priority ranking approach (must-see vs. nice-to-have)

                Be specific and practical (max 250 words).
                """
        return self.generate_response(prompt, max_tokens=350)
    
    def enhance_talk_descriptions(self, talks: List[Dict]) -> List[Dict]:
        """Add LLM-generated insights to talk descriptions"""
        if not self.available or len(talks) == 0:
            return talks
        
        # Process talks in batches to avoid overwhelming the LLM
        batch_size = 3
        enhanced_talks = []
        
        for i in range(0, len(talks), batch_size):
            batch = talks[i:i + batch_size]
            
            # Create prompt for batch
            talk_info = []
            for j, talk in enumerate(batch):
                talk_info.append(f"""
                                Talk {j+1}:
                                Title: {talk.get('title', 'Unknown')}
                                Speaker: {talk.get('speaker', 'Unknown')}
                                Abstract: {talk.get('abstract', 'No abstract')[:200]}...
                                """)
            
            prompt = f"""
                        You are a Monte Carlo methods expert analyzing conference talks. For each talk below, provide:
                        1. Key technical concepts (2-3 keywords)
                        2. Difficulty level (Beginner/Intermediate/Advanced)
                        3. Practical applications mentioned
                        4. One sentence summary of the main contribution

                        Talks to analyze:
                        {''.join(talk_info)}

                        Format your response as:
                        Talk 1: [Key concepts] | [Level] | [Applications] | [Summary]
                        Talk 2: [Key concepts] | [Level] | [Applications] | [Summary]
                        ...
                        """
            
            response = self.generate_response(prompt, max_tokens=500)
            
            # Parse response and enhance talks
            lines = response.split('\n')
            for j, talk in enumerate(batch):
                enhanced_talk = talk.copy()
                
                # Find corresponding line in response
                for line in lines:
                    if f"Talk {j+1}:" in line:
                        parts = line.split('|')
                        if len(parts) >= 4:
                            enhanced_talk['llm_concepts'] = parts[0].split(':')[1].strip()
                            enhanced_talk['llm_level'] = parts[1].strip()
                            enhanced_talk['llm_applications'] = parts[2].strip()
                            enhanced_talk['llm_summary'] = parts[3].strip()
                        break
                
                enhanced_talks.append(enhanced_talk)
        
        return enhanced_talks
    
    def generate_program_summary(self, selected_talks: List[Dict], user_interests: Dict) -> str:
        """Generate an intelligent summary of the selected program"""
        if not self.available:
            return "LLM summary unavailable - Ollama service not running."
        
        # Create overview of selected talks
        talk_summaries = []
        for talk in selected_talks[:10]:  # Limit to first 10 for brevity
            talk_summaries.append(f"- {talk.get('title', 'Unknown')}: {talk.get('llm_summary', talk.get('abstract', '')[:100])}")
        
        prompt = f"""
                    You are creating a personalized conference program summary for a researcher.

                    User's Research Focus:
                    - Keywords: {', '.join(user_interests.get('keywords', []))}
                    - Areas: {', '.join(user_interests.get('areas', []))}

                    Selected Talks ({len(selected_talks)} total):
                    {chr(10).join(talk_summaries[:10])}
                    {'...' if len(selected_talks) > 10 else ''}

                    Create a compelling program summary that includes:
                    1. Overview of the thematic focus
                    2. Key learning opportunities
                    3. Potential research collaborations/networking
                    4. How this selection advances their research goals
                    5. 2-3 specific action items for after the conference

                    Write in an encouraging, professional tone (max 400 words).
                    """
        
        return self.generate_response(prompt, max_tokens=500)

    def evaluate_talk_relevance(self, talk_data: Dict, user_interests: Dict) -> Dict:
        """Use AI to evaluate talk relevance based on user preferences"""
        if not self.available:
            return {'score': 50.0, 'reasoning': 'AI service unavailable - using default score'}
        
        # Create cache key based on talk title and user interests
        cache_key = f"{talk_data.get('title', '')[:50]}_{hash(str(sorted(user_interests.items())))}"
        
        # Check cache first
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        # Prepare talk information
        title = talk_data.get('title', 'Unknown Title')
        abstract = talk_data.get('abstract', talk_data.get('description', ''))[:500]  # Limit length
        speaker = talk_data.get('speaker', 'Unknown Speaker')
        talk_type = talk_data.get('type', 'contributed')
        
        # Get organizers info
        organizers_text = ''
        organizers = talk_data.get('organizers', [])
        if isinstance(organizers, list):
            organizers_text = ', '.join([org.get('name', '') if isinstance(org, dict) else str(org) for org in organizers])
        elif isinstance(organizers, str):
            organizers_text = organizers
        
        # Get schedule info
        schedule_info = talk_data.get('schedule_info', {})
        time_info = schedule_info.get('time', 'TBD')
        
        prompt = f"""
        You are an expert conference advisor helping a researcher evaluate talk relevance.

        RESEARCHER PROFILE:
        - Keywords: {', '.join(user_interests.get('keywords', []))}
        - Research Areas: {', '.join(user_interests.get('areas', []))}
        - Experience Level: {user_interests.get('experience', 'Not specified')}
        - Specific Preferences: {user_interests.get('preferences', 'None specified')}

        TALK TO EVALUATE:
        - Title: {title}
        - Speaker: {speaker}
        - Type: {talk_type}
        - Organizers: {organizers_text if organizers_text else 'N/A'}
        - Scheduled Time: {time_info}
        - Abstract: {abstract[:300]}...

        EVALUATION CRITERIA (in order of priority):
        1. **PREFERENCES COMPLIANCE**: If specific preferences are stated (time constraints, organizer preferences, content preferences), these are the highest priority
        2. **KEYWORD ALIGNMENT**: How well the talk content matches their research keywords
        3. **RESEARCH AREA FIT**: Relevance to their stated research areas
        4. **EXPERIENCE APPROPRIATENESS**: Suitable for their experience level

        **IMPORTANT RULES**:
        - If preferences mention excluding certain times/organizers/topics, score this talk 0-20 if it matches those exclusions
        - If preferences mention favoring certain organizers/topics, boost the score significantly (add 20-30 points) if it matches
        - Consider time constraints mentioned in preferences when evaluating scheduled time

        RESPONSE FORMAT:
        Score: [0-100 integer score where 0=completely irrelevant/excluded, 100=perfect match]
        Reasoning: [2-3 sentence explanation focusing on how this talk aligns with stated preferences and research interests]
        """
        
        try:
            response = self.generate_response(prompt, max_tokens=200)
            
            # Parse the response
            lines = response.strip().split('\n')
            score = 50.0  # default
            reasoning = "Unable to parse AI response"
            
            for line in lines:
                if line.lower().startswith('score:'):
                    try:
                        score_text = line.split(':', 1)[1].strip()
                        # Extract number from text that might contain extra words
                        import re
                        score_match = re.search(r'\b(\d+)\b', score_text)
                        if score_match:
                            score = min(float(score_match.group(1)), 100.0)
                    except:
                        pass
                elif line.lower().startswith('reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            result = {'score': score, 'reasoning': reasoning}
            
            # Cache the result
            self.evaluation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"Error in AI relevance evaluation: {e}")
            result = {'score': 50.0, 'reasoning': f'AI evaluation failed: {str(e)}'}
            self.evaluation_cache[cache_key] = result
            return result
    
    def evaluate_talks_batch(self, talks: List[Dict], user_interests: Dict) -> List[Dict]:
        """Evaluate multiple talks in a single AI request for better performance"""
        if not talks:
            return []
        
        # Prepare batch prompt
        keywords_str = ', '.join(user_interests.get('keywords', []))
        areas_str = ', '.join(user_interests.get('areas', []))
        preferences = user_interests.get('preferences', '')
        experience = user_interests.get('experience', '')
        
        # Create a concise summary of each talk for batch processing
        talks_summary = []
        for i, talk in enumerate(talks):
            title = talk.get('title', 'Unknown')
            speaker = talk.get('speaker', 'Unknown')
            abstract = talk.get('abstract', '')[:200]  # Limit length
            talks_summary.append(f"Talk {i+1}: '{title}' by {speaker}. Abstract: {abstract}...")
        
        prompt = f"""
You are evaluating {len(talks)} conference talks for relevance to a researcher's interests.

RESEARCHER PROFILE:
- Keywords: {keywords_str}
- Research Areas: {areas_str}
- Experience Level: {experience}
- Additional Preferences: {preferences}

TALKS TO EVALUATE:
{chr(10).join(talks_summary)}

EVALUATION CRITERIA (in order of priority):
1. **PREFERENCES COMPLIANCE**: If the researcher has stated specific preferences (time constraints, organizer preferences, content preferences), these take precedence
2. **KEYWORD ALIGNMENT**: How well the talk content matches their research keywords
3. **RESEARCH AREA FIT**: Relevance to their stated research areas
4. **EXPERIENCE APPROPRIATENESS**: Suitable for their experience level

For each talk, provide a score (0-100) and brief reasoning. Format:
Talk 1: Score: [number] | Reasoning: [brief explanation focusing on preference alignment and keyword matches]
Talk 2: Score: [number] | Reasoning: [brief explanation focusing on preference alignment and keyword matches]
...

**IMPORTANT**: If preferences mention excluding certain organizers/times/topics, score those talks very low (0-20). If preferences mention favoring certain organizers/topics, boost those scores significantly.
"""
        
        try:
            response = self.generate_response(prompt, max_tokens=800)
            return self._parse_batch_response(response, talks)
            
        except Exception as e:
            print(f"Error in batch AI evaluation: {e}")
            # Fallback to individual evaluations
            results = []
            for talk in talks:
                individual_result = self.evaluate_talk_relevance(talk, user_interests)
                results.append({
                    'talk': talk,
                    'score': individual_result.get('score', 50.0),
                    'reasoning': individual_result.get('reasoning', 'Individual fallback evaluation')
                })
            return results
    
    def _parse_batch_response(self, response: str, talks: List[Dict]) -> List[Dict]:
        """Parse the batch AI response into individual talk results"""
        results = []
        lines = response.strip().split('\n')
        
        import re
        
        for i, talk in enumerate(talks):
            score = 50.0  # default
            reasoning = "Unable to parse AI response"
            
            # Look for lines that match "Talk X: Score: Y | Reasoning: Z"
            pattern = f"Talk {i+1}:"
            for line in lines:
                if pattern in line:
                    # Extract score
                    score_match = re.search(r'Score:\s*(\d+)', line)
                    if score_match:
                        score = min(float(score_match.group(1)), 100.0)
                    
                    # Extract reasoning
                    reasoning_match = re.search(r'Reasoning:\s*(.+)', line)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                    break
            
            results.append({
                'talk': talk,
                'score': score,
                'reasoning': reasoning
            })
        
        return results
    
    def clear_evaluation_cache(self):
        """Clear the evaluation cache (useful when switching models or interests)"""
        self.evaluation_cache.clear()
        print("AI evaluation cache cleared")
    
    def calculate_optimal_batch_size(self, filtered_talks: List[Dict]) -> int:
        """Calculate optimal batch size based on dataset characteristics"""
        if not self.available:
            return len(filtered_talks)  # Process all at once in fallback mode
        
        config = self.batch_config
        num_talks = len(filtered_talks)
        
        # Base batch size selection
        if num_talks <= 20:
            batch_size = config['default_small']
        elif num_talks <= 100:
            batch_size = config['default_medium']
        else:
            batch_size = config['default_large']
        
        # Adjust based on content complexity
        if num_talks > 0:
            sample_size = min(10, num_talks)
            avg_content_length = sum(
                len(talk.get('abstract', '') + talk.get('title', '')) 
                for talk in filtered_talks[:sample_size]
            ) / sample_size
            
            if avg_content_length > config['content_length_threshold']:
                # Long content: reduce batch size for better AI focus
                batch_size = max(config['min_batch_size'], batch_size - 2)
            elif avg_content_length < 200:
                # Short content: can handle larger batches
                batch_size = min(config['max_batch_size'], batch_size + 3)
        
        # Ensure within bounds
        batch_size = max(config['min_batch_size'], min(config['max_batch_size'], batch_size))
        
        return batch_size

    def select_best_parallel_session(self, parallel_sessions: List[Dict], user_interests: Dict, time_slot: str) -> Dict:
        """Use AI to select the best session from parallel options based on rich data."""
        if not self.available or not parallel_sessions:
            # Fallback: return first session if AI is unavailable
            return parallel_sessions[0] if parallel_sessions else None

        # Prepare detailed session summaries for AI evaluation
        session_summaries = []
        for i, session in enumerate(parallel_sessions):
            session_id = session.get('id', f'Session_{i+1}')
            title = session.get('title', 'Unknown Session')
            
            # Handle sessions with full LaTeX content
            if session.get('latex_content'):
                latex_content = session['latex_content']
                talk_count = session.get('talk_count', 0)
                
                # Extract a clean summary of the session content for AI analysis
                # Remove excessive LaTeX formatting but keep the essential content
                clean_content = re.sub(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', '', latex_content, flags=re.DOTALL)
                clean_content = re.sub(r'\\item\[\{.*?\}\].*?(?=\\item|\Z)', '', clean_content, flags=re.DOTALL)
                clean_content = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*', ' ', clean_content)
                clean_content = re.sub(r'[{}\\]', ' ', clean_content)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                
                # Truncate if too long (keep first 2000 chars to include multiple talks)
                if len(clean_content) > 2000:
                    clean_content = clean_content[:2000] + "... [content truncated]"
                
                session_summaries.append(f"""
Session {i+1} ({session_id}):
Title: {title}
Number of Talks: {talk_count}
Full LaTeX Content Available: YES

Session Content:
{clean_content}
""")
                
            else:
                # Fallback mode for sessions without LaTeX content
                description = session.get('description', '')
                organizers = session.get('organizers', [])
                organizer_names = []

                if isinstance(organizers, list):
                    organizer_names = [org.get('name', '') if isinstance(org, dict) else str(org) for org in organizers]
                elif isinstance(organizers, str):
                    organizer_names = [organizers]

                # Include basic talks info from parsed data
                talks_info = ""
                if session.get('talks'):
                    talk_details = []
                    for talk in session['talks']:
                        talk_title = talk.get('title', 'No Title')
                        talk_speaker = talk.get('speaker', 'No Speaker')
                        talk_abstract = talk.get('abstract', 'No Abstract')
                        # Provide a good chunk of the abstract for the AI to analyze
                        talk_details.append(f"  - Talk: '{talk_title}' by {talk_speaker}. Abstract: {talk_abstract[:250]}...")
                    if talk_details:
                        talks_info = "Talks in this session:\n" + "\n".join(talk_details)

                session_summaries.append(f"""
Session {i+1} ({session_id}):
Title: {title}
Organizers: {', '.join(organizer_names)}
Description: {description}
Full LaTeX Content Available: NO (using fallback data)
{talks_info}
""")

        prompt = f"""
You are an expert conference advisor helping a researcher choose the best session to attend from parallel sessions.

RESEARCHER PROFILE:
- Keywords: {', '.join(user_interests.get('keywords', []))}
- Research Areas: {', '.join(user_interests.get('areas', []))}
- Experience Level: {user_interests.get('experience', 'Not specified')}
- Preferences: {user_interests.get('preferences', 'None specified')}

TIME SLOT: {time_slot}

PARALLEL SESSIONS TO CHOOSE FROM:
{chr(10).join(session_summaries)}

IMPORTANT ANALYSIS INSTRUCTIONS:
1. **Full LaTeX Content**: Some sessions include complete LaTeX content from their original session files, giving you access to comprehensive talk abstracts, mathematical content, and detailed descriptions.
2. **Deep Content Analysis**: When LaTeX content is available, analyze the full mathematical content, abstracts, and technical details of each talk within the session.
3. **Keyword Matching**: Look for the researcher's keywords not just in session titles, but throughout all talk abstracts and descriptions.
4. **Technical Relevance**: Consider the technical depth and mathematical content when matching to the researcher's experience level and research areas.
5. **Select ONE Best Session**: Choose the single session that offers maximum value based on content relevance, not just title similarity.

RESPONSE FORMAT (respond EXACTLY in this format with NO additional text):
Best Session: [Session number 1-{len(parallel_sessions)}]
Reasoning: [2-3 sentences explaining why this session is the best choice, referencing specific talks, mathematical content, or technical aspects that influenced your decision.]
"""
        
        try:
            #print(f"DEBUG: AI prompt for {time_slot} is {len(prompt)} chars long.")
            response = self.generate_response(prompt, max_tokens=300)
            
            if IS_DEBUG_APP:
                print(f"DEBUG: Raw AI response for {time_slot}: {response}")
            
            # Clean up the response by removing <think> tags and any content within them
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            if IS_DEBUG_APP:
                print(f"DEBUG: Cleaned AI response: {response}")
            
            # Parse the response with more robust parsing
            lines = response.strip().split('\n')
            selected_session_idx = 0  # default to first session
            reasoning = "Default selection - AI response parsing failed"
            
            # Try multiple parsing strategies
            full_text = response.lower()
            
            # Strategy 1: Look for "best session:" pattern
            for line in lines:
                line_lower = line.lower().strip()
                if line_lower.startswith('best session:'):
                    try:
                        session_text = line.split(':', 1)[1].strip()
                        session_match = re.search(r'\b(\d+)\b', session_text)
                        if session_match:
                            session_num = int(session_match.group(1))
                            if 1 <= session_num <= len(parallel_sessions):
                                selected_session_idx = session_num - 1
                    except:
                        pass
                elif line_lower.startswith('reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            # Strategy 2: If parsing failed, try to find session number and reasoning independently
            if reasoning == "Default selection - AI response parsing failed":
                session_match = re.search(r'best session\s*[:\s-]*\s*(\d+)', full_text)
                if session_match:
                    session_num = int(session_match.group(1))
                    if 1 <= session_num <= len(parallel_sessions):
                        selected_session_idx = session_num - 1
                
                reasoning_text = self.extract_meaningful_reasoning(response)
                if reasoning_text:
                    reasoning = reasoning_text
                else:
                    reasoning = "AI selected this session based on user interests."

            selected_session = parallel_sessions[selected_session_idx].copy()
            selected_session['ai_selection_reasoning'] = reasoning
            
            if IS_DEBUG_APP:
                print(f"DEBUG: AI selected session {selected_session_idx + 1} for {time_slot}: {selected_session.get('title', 'Unknown')}")
                print(f"DEBUG: Reasoning: {reasoning}")
            
            return selected_session
            
        except Exception as e:
            print(f"Error in AI session selection: {e}")
            fallback_session = parallel_sessions[0].copy() if parallel_sessions else None
            if fallback_session:
                fallback_session['ai_selection_reasoning'] = f'AI selection failed: {str(e)}, using first session as fallback'
            return fallback_session
    
    def extract_meaningful_reasoning(self, response: str) -> str:
        """Extract meaningful reasoning from AI response even if format is imperfect"""
        if not response:
            return ""
        
        # Clean the response
        response = response.strip()
        
        # Look for sentences that explain reasoning
        reasoning_indicators = [
            'because', 'since', 'as', 'given', 'considering', 'due to',
            'focuses on', 'specializes in', 'covers', 'addresses',
            'relevant to', 'matches', 'aligns with', 'suitable for'
        ]
        
        sentences = response.split('.')
        best_reasoning = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Skip the "Best Session: X" line
            if re.search(r'best session\s*:\s*\d+', sentence, re.IGNORECASE):
                continue
                
            # Look for sentences with reasoning indicators
            sentence_lower = sentence.lower()
            for indicator in reasoning_indicators:
                if indicator in sentence_lower:
                    # This sentence likely contains reasoning
                    if len(sentence) > len(best_reasoning):
                        best_reasoning = sentence
                    break
        
        # If no reasoning indicators found, use the longest substantive sentence
        if not best_reasoning:
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 30 and 
                    not re.search(r'best session\s*:\s*\d+', sentence, re.IGNORECASE) and
                    not sentence.lower().startswith('reasoning:')):
                    if len(sentence) > len(best_reasoning):
                        best_reasoning = sentence
        
        # Clean up the reasoning text
        if best_reasoning:
            # Remove any remaining session references at the start
            best_reasoning = re.sub(r'^[^a-zA-Z]*session\s+\d+[^a-zA-Z]*', '', best_reasoning, flags=re.IGNORECASE)
            best_reasoning = best_reasoning.strip()
            
            # Ensure it doesn't start with punctuation
            if best_reasoning.startswith((',', ':', ';', '-')):
                best_reasoning = best_reasoning[1:].strip()
            
            # Capitalize first letter
            if best_reasoning and best_reasoning[0].islower():
                best_reasoning = best_reasoning[0].upper() + best_reasoning[1:]
            
            # Limit length
            if len(best_reasoning) > 200:
                best_reasoning = best_reasoning[:197] + "..."
        
        return best_reasoning if best_reasoning else ""


class MCMStreamlitApp:
    def __init__(self):
        # Set page config
        st.set_page_config(
            page_title="MCM 2025: Personalized Program Book Generator",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize LLM assistant
        if 'llm_assistant' not in st.session_state:
            # Get initial model from session state or use default
            initial_model = st.session_state.get('selected_model', 'qwen3')
            with st.spinner("Initializing AI assistant..."):
                st.session_state.llm_assistant = LLMAssistant(model_name=initial_model)
        
        self.llm = st.session_state.llm_assistant
        
        # Initialize session state
        if 'abstracts_data' not in st.session_state:
            st.session_state.abstracts_data = {}
        if 'schedule_data' not in st.session_state:
            st.session_state.schedule_data = []
        if 'current_talks' not in st.session_state:
            st.session_state.current_talks = []
        if 'selected_talks' not in st.session_state:
            st.session_state.selected_talks = []
        if 'user_interests' not in st.session_state:
            st.session_state.user_interests = {}
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = 'qwen3'
        
        # Paths - Fixed path resolution
        # App is in preprocess/app/, so we need to go back to preprocess/ directory
        app_dir = Path(__file__).parent.resolve()  # This is preprocess/app/
        self.base_path = app_dir.parent  # This is preprocess/
        self.abstracts_path = self.base_path / "input" / "abstracts"
        self.schedule_path = self.base_path / "interim" / "schedule_joined.csv"
        self.contrib_talks_path = self.base_path / "interim" / "contributed_talk_submissions_talkid.csv"
        self.special_sessions_path = self.base_path / "interim" / "special_session_abstracts_talkid.csv" # special_session_submissions_sessionid.csv
        self.plenary_sessions_path = self.base_path / "interim" / "plenary_abstracts_talkid.csv"
        # Change output path to MCM_ProgramBook_TEX directory
        self.output_path = self.base_path.parent / "MCM_ProgramBook_TEX"
        # Set tex_dir to the same location for plenary session file access
        self.tex_dir = self.output_path
        
        # Debug: Print paths to verify
        if IS_DEBUG_APP:
            print(f"DEBUG: App directory: {app_dir}")
            print(f"DEBUG: Base path: {self.base_path}")
            print(f"DEBUG: Abstracts path: {self.abstracts_path}")
            print(f"DEBUG: Schedule path: {self.schedule_path}")
            print(f"DEBUG: Abstracts exists: {self.abstracts_path.exists()}")
            print(f"DEBUG: Schedule exists: {self.schedule_path.exists()}")
        
        # Create output directory
        self.output_path.mkdir(exist_ok=True)
        
        # Load data on first run
        if not st.session_state.data_loaded:
            self.load_data()
    
    def load_data(self):
        """Load conference data from files"""
        try:
            with st.spinner("Loading conference data..."):
                # Initialize counters
                schedule_count = 0
                abstracts_count = 0
                
                # Load schedule data
                if self.schedule_path.exists():
                    schedule_df = pd.read_csv(self.schedule_path)
                    st.session_state.schedule_data = schedule_df.to_dict('records')
                    schedule_count = len(st.session_state.schedule_data)
                    print(f"Loaded {schedule_count} schedule items from {self.schedule_path}")
                else:
                    st.session_state.schedule_data = []
                    st.warning(f"Schedule file not found: {self.schedule_path}")
                    print(f"ERROR: Schedule file not found at {self.schedule_path}")
                
                # Load abstracts
                self.load_abstracts()
                abstracts_count = len(st.session_state.abstracts_data)
                print(f"Loaded {abstracts_count} abstracts")
                
                # Load contributed talks metadata
                if self.contrib_talks_path.exists():
                    contrib_df = pd.read_csv(self.contrib_talks_path)
                    st.session_state.contrib_talks_data = contrib_df.to_dict('records')
                    print(f"Loaded {len(st.session_state.contrib_talks_data)} contributed talks")
                else:
                    st.session_state.contrib_talks_data = []
                
                # Load special sessions metadata
                if self.special_sessions_path.exists():
                    sessions_df = pd.read_csv(self.special_sessions_path)
                    st.session_state.special_sessions_data = sessions_df.to_dict('records')
                    print(f"Loaded {len(st.session_state.special_sessions_data)} special sessions")
                else:
                    st.session_state.special_sessions_data = []
                
                st.session_state.data_loaded = True
                
                # Show detailed loading summary
                total_items = schedule_count + abstracts_count
                if not (total_items > 0):
                #    st.success(f"✅ Data loaded successfully: {schedule_count} schedule items + {abstracts_count} abstracts = #{total_items} total items")
                #else:
                    st.error("❌ No data was loaded. Please check that data files exist in the correct locations.")
                    # Show file paths for debugging
                    st.write("Expected file paths:")
                    st.write(f"- Schedule: {self.schedule_path}")
                    st.write(f"- Abstracts: {self.abstracts_path}")
                    st.write(f"- Contrib talks: {self.contrib_talks_path}")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            import traceback
            st.error(f"Full error: {traceback.format_exc()}")
    
    def load_abstracts(self):
        """Load all conference data from the single consolidated LaTeX file"""
        # Try to load from consolidated file first
        consolidated_file = self.base_path / "output" / "MCM2025_consolidated.tex"
        
        if consolidated_file.exists():
            print(f"Loading from consolidated file: {consolidated_file}")
            self.load_consolidated_data(consolidated_file)
            return
        
        # Fallback to original method if consolidated file doesn't exist
        if IS_DEBUG_APP:
            print(f"📁 Using individual LaTeX files (consolidated file not found - this is normal)")
            #print(f"DEBUG: Checking abstracts path: {self.abstracts_path}")
            #print(f"DEBUG: Abstracts path exists: {self.abstracts_path.exists()}")
        
        loaded_count = 0
        error_count = 0
        
        # First, load from the main LaTeX files in MCM_ProgramBook_TEX/
        main_latex_files = [
            self.base_path.parent / "MCM_ProgramBook_TEX" / "contributed_talk_submissions_talks.tex",
            self.base_path.parent / "MCM_ProgramBook_TEX" / "plenary_abstracts_talks.tex", 
            self.base_path.parent / "MCM_ProgramBook_TEX" / "special_session_abstracts_talks.tex",
            self.base_path.parent / "MCM_ProgramBook_TEX" / "special_session_submissions_talks.tex"
        ]
        
        for latex_file in main_latex_files:
            if latex_file.exists():
                print(f"Loading talks from: {latex_file.name}")
                try:
                    content = self.safe_read_file(latex_file)
                    if not content:
                        continue
                    
                    # Extract all talks from this file
                    talk_matches = re.finditer(r'\\begin\{talk\}(.*?)\\end\{talk\}', content, re.DOTALL)
                    
                    for match in talk_matches:
                        talk_content = match.group(0)  # Include \begin{talk} and \end{talk}
                        
                        # Extract talk ID from the content
                        id_match = re.search(r'\{([^}]*)\}%\s*\[8\]\s*talk id', talk_content)
                        if id_match:
                            talk_id = id_match.group(1).strip()
                            print(f"  Found talk: {talk_id}")
                            
                            # Parse the talk content
                            abstract_info = self.parse_talk_from_main_file(talk_content, talk_id)
                            if abstract_info:
                                st.session_state.abstracts_data[talk_id] = abstract_info
                                loaded_count += 1
                                print(f"  Successfully loaded: {talk_id}")
                            else:
                                print(f"  Failed to parse: {talk_id}")
                                error_count += 1
                        else:
                            print(f"  Could not extract ID from talk")
                            error_count += 1
                            
                except Exception as e:
                    print(f"Error loading {latex_file}: {e}")
                    error_count += 1
            else:
                print(f"File not found: {latex_file}")
        
        # Then load sessions from special_session_submissions_talks.tex 
        session_file = self.base_path.parent / "MCM_ProgramBook_TEX" / "special_session_submissions_talks.tex"
        if session_file.exists():
            print(f"Loading sessions from: {session_file.name}")
            try:
                content = self.safe_read_file(session_file)
                if not content:
                    print(f"Failed to read content from {session_file}")
                else:
                    # Extract all sessions from this file
                    session_matches = re.finditer(r'\\begin\{session\}(.*?)\\end\{session\}', content, re.DOTALL)
                
                for match in session_matches:
                    session_content = match.group(0)  # Include \begin{session} and \end{session}
                    
                    # Extract session ID from the content  
                    id_match = re.search(r'\{([^}]*)\}%\s*\[8\]\s*session id', session_content)
                    if id_match:
                        session_id = id_match.group(1).strip()
                        print(f"  Found session: {session_id}")
                        
                        # Parse the session content
                        session_info = self.parse_session_from_main_file(session_content, session_id)
                        if session_info:
                            st.session_state.abstracts_data[session_id] = session_info
                            loaded_count += 1
                            print(f"  Successfully loaded session: {session_id}")
                        else:
                            print(f"  Failed to parse session: {session_id}")
                            error_count += 1
                    else:
                        print(f"  Could not extract ID from session")
                        error_count += 1
                        
            except Exception as e:
                print(f"Error loading sessions from {session_file}: {e}")
                error_count += 1
        
        # Finally, load any additional individual files from preprocess/input/abstracts/ 
        # (for compatibility with existing structure)
        if self.abstracts_path.exists():
            abstract_files = list(self.abstracts_path.glob("*.tex"))
            print(f"Found {len(abstract_files)} individual abstract files")
            
            for abstract_file in abstract_files:
                try:
                    # Skip if already loaded from main files
                    if abstract_file.stem in st.session_state.abstracts_data:
                        continue
                        
                    content = self.safe_read_file(abstract_file)
                        
                    # Parse the abstract content
                    abstract_info = self.parse_abstract(content, abstract_file.stem)
                    if abstract_info:
                        st.session_state.abstracts_data[abstract_file.stem] = abstract_info
                        loaded_count += 1
                        print(f"Successfully loaded individual file: {abstract_file.stem}")
                    else:
                        print(f"Failed to parse individual file: {abstract_file.stem}")
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    print(f"Error loading individual file {abstract_file}: {e}")
        else:
            print(f"Individual abstracts directory not found: {self.abstracts_path}")
        
        # Load session room/time information from sessXX.tex files
        self.load_session_room_info()
        
        print(f"Successfully loaded {loaded_count} abstracts total, {error_count} errors")
    
    def load_session_room_info(self):
        """Load room and time information from sessXX.tex files"""
        mcm_path = self.base_path.parent / "MCM_ProgramBook_TEX"
        
        # Load plenary sessions (P1.tex, P2.tex, etc.)
        for p_file in mcm_path.glob("P*.tex"):
            try:
                room_info = self.extract_session_room_from_file(p_file)
                if room_info:
                    self.update_talk_with_room_info(room_info)
            except Exception as e:
                print(f"Error loading room info from {p_file}: {e}")
        
        # Load special/technical sessions (sessS*.tex, sessT*.tex)
        for sess_file in mcm_path.glob("sess*.tex"):
            try:
                room_info = self.extract_session_room_from_file(sess_file)
                if room_info:
                    self.update_talk_with_room_info(room_info)
            except Exception as e:
                print(f"Error loading room info from {sess_file}: {e}")
    
    def extract_session_room_from_file(self, file_path: Path) -> Dict:
        """Extract room and time information from session files"""
        try:
            content = self.safe_read_file(file_path)
            if not content:
                return {}
            
            room_info = {}
            
            # Handle plenary sessions with \tablePlenary format
            plenary_pattern = r'\\tablePlenary\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}'
            plenary_matches = re.findall(plenary_pattern, content)
            
            for match in plenary_matches:
                time, room, chair, speaker, title, talk_id = match
                room_info[talk_id.strip()] = {
                    'time': time.strip(),
                    'room': room.strip(),
                    'chair': chair.strip(),
                    'speaker': speaker.strip(),
                    'title': title.strip()
                }
                if IS_DEBUG_APP:
                    print(f"DEBUG: Plenary {talk_id} -> Time: {time.strip()}, Room: {room.strip()}")
            
            # Handle special/technical sessions with \timeslot format
            timeslot_pattern = r'\\timeslot\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}'
            timeslot_matches = re.findall(timeslot_pattern, content)
            
            for match in timeslot_matches:
                time_desc, start_time, end_time, room = match
                
                # Find all talks in this session
                session_talks = re.findall(r'\\sessionTalk\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}', content)
                
                for talk_match in session_talks:
                    talk_title, speaker, talk_id = talk_match
                    full_time = f"{time_desc.strip()}, {start_time.strip()}–{end_time.strip()}"
                    
                    room_info[talk_id.strip()] = {
                        'time': full_time,
                        'room': room.strip(),
                        'chair': 'TBD',
                        'speaker': speaker.strip(),
                        'title': talk_title.strip()
                    }
                    if IS_DEBUG_APP:
                        print(f"DEBUG: Session talk {talk_id} -> Time: {full_time}, Room: {room.strip()}")
            
            return room_info
            
        except Exception as e:
            print(f"Error extracting room info from {file_path}: {e}")
            return {}
    
    def update_talk_with_room_info(self, room_info: Dict):
        """Update talks in session state with room information"""
        for talk_id, info in room_info.items():
            # Try to find matching talk in abstracts_data
            for abstract_id, abstract_data in st.session_state.abstracts_data.items():
                # Match by talk ID or by title/speaker
                if (abstract_id == talk_id or 
                    talk_id in abstract_id or 
                    abstract_id in talk_id or
                    (abstract_data.get('title', '').strip() == info.get('title', '').strip() and
                     abstract_data.get('speaker', '').strip() == info.get('speaker', '').strip())):
                    
                    # Update schedule info
                    if 'schedule_info' not in abstract_data:
                        abstract_data['schedule_info'] = {}
                    
                    abstract_data['schedule_info'].update({
                        'time': info.get('time'),
                        'room': info.get('room'),
                        'chair': info.get('chair')
                    })
                    
                    # Also update time_slot field for backward compatibility
                    if not abstract_data.get('time_slot'):
                        abstract_data['time_slot'] = info.get('time')
                    
                    if IS_DEBUG_APP:
                        print(f"DEBUG: Updated {abstract_id} with room info: {info.get('room')}, time: {info.get('time')}")
                    break
    
    def safe_read_file(self, file_path, encoding='utf-8'):
        """Safely read a file with encoding fallback to handle problematic characters"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                # Ensure the content is properly encoded
                return self.safe_text_encode(content)
        except UnicodeDecodeError:
            # Fallback to latin1 which can decode any byte sequence
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    content = f.read()
                    # Ensure the content is properly encoded
                    return self.safe_text_encode(content)
            except Exception as e:
                print(f"ERROR: Failed to read file {file_path} with both UTF-8 and Latin1: {e}")
                return ""
        except Exception as e:
            print(f"ERROR: Failed to read file {file_path}: {e}")
            return ""

    def safe_text_encode(self, text):
        """Safely encode text to handle problematic characters"""
        if not text:
            return ""
        try:
            # Try to normalize the text to handle any encoding issues
            if isinstance(text, bytes):
                # If it's bytes, decode it safely
                text = text.decode('utf-8', errors='replace')
            
            # Ensure clean UTF-8 encoding
            text = text.encode('utf-8', errors='replace').decode('utf-8')
            return text
        except Exception as e:
            print(f"ERROR: Failed to safely encode text: {e}")
            return str(text) if text else ""

    def load_consolidated_data(self, consolidated_file: Path):
        """Load all conference data from the single consolidated LaTeX file"""
        try:
            content = self.safe_read_file(consolidated_file)
            if not content:
                return
            
            print(f"Consolidated file loaded: {len(content)} characters")
            
            # Parse all talks from the consolidated content
            self.parse_consolidated_content(content)
            
        except Exception as e:
            print(f"Error loading consolidated file: {e}")
            st.error(f"Error loading consolidated file: {e}")

    def parse_consolidated_content(self, content: str):
        """Parse all content from the consolidated LaTeX file"""
        loaded_count = 0
        error_count = 0
        
        # Extract all talks
        talk_matches = list(re.finditer(r'\\begin\{talk\}(.*?)\\end\{talk\}', content, re.DOTALL))
        print(f"Found {len(talk_matches)} talks in consolidated file")
        
        for match in talk_matches:
            try:
                talk_content = match.group(0)  # Include \begin{talk} and \end{talk}
                
                # Extract talk ID from the content
                id_match = re.search(r'\{([^}]*)\}%\s*\[8\]\s*talk id', talk_content)
                if id_match:
                    talk_id = id_match.group(1).strip()
                    print(f"  Processing talk: {talk_id}")
                    
                    # Parse the talk content
                    abstract_info = self.parse_talk_from_consolidated(talk_content, talk_id)
                    if abstract_info:
                        st.session_state.abstracts_data[talk_id] = abstract_info
                        loaded_count += 1
                        print(f"  ✅ Successfully loaded: {talk_id}")
                    else:
                        print(f"  ❌ Failed to parse: {talk_id}")
                        error_count += 1
                else:
                    print(f"  ❌ Could not extract ID from talk")
                    error_count += 1
                    
            except Exception as e:
                print(f"Error processing talk: {e}")
                error_count += 1
        
        # Extract all sessions
        session_matches = list(re.finditer(r'\\begin\{session\}(.*?)\\end\{session\}', content, re.DOTALL))
        print(f"Found {len(session_matches)} sessions in consolidated file")
        
        for match in session_matches:
            try:
                session_content = match.group(0)  # Include \begin{session} and \end{session}
                
                # Extract session ID from the content
                id_match = re.search(r'\{([^}]*)\}%\s*\[8\]\s*session id', session_content)
                if id_match:
                    session_id = id_match.group(1).strip()
                    print(f"  Processing session: {session_id}")
                    
                    # Parse the session content
                    session_info = self.parse_session_from_consolidated(session_content, session_id)
                    if session_info:
                        st.session_state.abstracts_data[session_id] = session_info
                        loaded_count += 1
                        print(f"  ✅ Successfully loaded session: {session_id}")
                    else:
                        print(f"  ❌ Failed to parse session: {session_id}")
                        error_count += 1
                else:
                    print(f"  ❌ Could not extract ID from session")
                    error_count += 1
                    
            except Exception as e:
                print(f"Error processing session: {e}")
                error_count += 1
        
        # Extract schedule information from the consolidated file
        self.parse_schedule_from_consolidated(content)
        
        print(f"Consolidated loading complete: {loaded_count} items loaded, {error_count} errors")
        st.success(f"✅ Loaded {loaded_count} items from consolidated file ({error_count} errors)")

    def parse_talk_from_consolidated(self, content: str, talk_id: str) -> Dict:
        """Parse talk content from consolidated LaTeX file"""
        if IS_DEBUG_APP:
            print(f"DEBUG: Parsing consolidated talk: {talk_id}")
        
        # Extract the parameters using the same logic as before
        talk_match = re.search(r'\\begin\{talk\}(.*?)\\end\{talk\}', content, re.DOTALL)
        
        if not talk_match:
            print(f"ERROR: Could not find talk environment in {talk_id}")
            return None
        
        talk_content_inner = talk_match.group(1).strip()
        
        # Extract the 9 parameters: [1] title, [2] speaker, [3] affiliation, [4] email, 
        # [5] coauthors, [6] special session, [7] time slot, [8] talk id, [9] session id
        params = self.extract_parameters_from_content(talk_content_inner, 9)
        
        # Extract parameters with defaults
        title = params[0] if len(params) > 0 else "Unknown Title"
        speaker = params[1] if len(params) > 1 else "Unknown Speaker" 
        affiliation = params[2] if len(params) > 2 else ""
        email = params[3] if len(params) > 3 else ""
        coauthors = params[4] if len(params) > 4 else ""
        special_session = params[5] if len(params) > 5 else ""
        time_slot = params[6] if len(params) > 6 else ""
        session_id = params[8] if len(params) > 8 else ""
        
        # Extract abstract text (everything after the parameters)
        abstract_text = self.extract_abstract_from_content(talk_content_inner, params)
        
        if IS_DEBUG_APP:
            print(f"DEBUG: Consolidated talk {talk_id} - Title: '{title}', Speaker: '{speaker}', Time: '{time_slot}'")
        
        return {
            'type': 'talk',
            'id': talk_id,
            'title': title,
            'speaker': speaker,
            'affiliation': affiliation,
            'email': email,
            'coauthors': coauthors,
            'special_session': special_session,
            'time_slot': time_slot,
            'session_id': session_id,
            'abstract': abstract_text,
            'content': content
        }

    def parse_session_from_consolidated(self, content: str, session_id: str) -> Dict:
        """Parse session content from consolidated LaTeX file"""
        if IS_DEBUG_APP:
            print(f"DEBUG: Parsing consolidated session: {session_id}")
        
        # Extract session parameters
        session_match = re.search(r'\\begin\{session\}(.*?)\\end\{session\}', content, re.DOTALL)
        
        if not session_match:
            print(f"ERROR: Could not find session environment in {session_id}")
            return None
        
        session_content_inner = session_match.group(1).strip()
        
        # Extract the 9 parameters for session format
        params = self.extract_parameters_from_content(session_content_inner, 9)
        
        # Extract parameters with defaults
        title = params[0] if len(params) > 0 else "Unknown Session"
        organizer1_name = params[1] if len(params) > 1 else ""
        organizer1_affiliation = params[2] if len(params) > 2 else ""
        organizer1_email = params[3] if len(params) > 3 else ""
        organizer2_name = params[4] if len(params) > 4 else ""
        organizer2_affiliation = params[5] if len(params) > 5 else ""
        organizer2_email = params[6] if len(params) > 6 else ""
        third_organizer = params[8] if len(params) > 8 else ""
        
        # Build organizers list
        organizers = []
        if organizer1_name:
            organizers.append({
                'name': organizer1_name,
                'affiliation': organizer1_affiliation,
                'email': organizer1_email
            })
        if organizer2_name:
            organizers.append({
                'name': organizer2_name,
                'affiliation': organizer2_affiliation,
                'email': organizer2_email
            })
        
        # Extract third organizer if present
        if third_organizer and third_organizer.startswith('\\thirdorganizer{'):
            third_match = re.search(r'\\thirdorganizer\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}', third_organizer)
            if third_match:
                organizers.append({
                    'name': third_match.group(1).strip(),
                    'affiliation': third_match.group(2).strip(),
                    'email': third_match.group(3).strip()
                })
        
        # Extract session description
        description = self.extract_description_from_content(session_content_inner, params)
        
        if IS_DEBUG_APP:
            print(f"DEBUG: Consolidated session {session_id} - Title: '{title}', Organizers: {len(organizers)}")
        
        return {
            'type': 'session',
            'id': session_id,
            'title': title,
            'organizers': organizers,
            'description': description,
            'content': content
        }

    def parse_schedule_from_consolidated(self, content: str):
        """Extract schedule/room information from the consolidated LaTeX file"""
        print("DEBUG: Parsing schedule information from consolidated file")
        
        # Look for schedule tables in the consolidated content
        # The schedule information is typically in \tableTalk commands
        schedule_pattern = r'\\tableTalk\{([^}]*)\}\s*\{([^}]*)\}\s*\{([^}]*)\}'  
        schedule_matches = re.findall(schedule_pattern, content)
        
        print(f"Found {len(schedule_matches)} schedule entries")
        
        # Also look for room assignments in table format
        # Find table rows with multiple \tableTalk entries
        table_row_pattern = r'\\tableTime\{[^}]*\}\{[^}]*\}(.*?)\\\\\\hline'
        table_rows = re.findall(table_row_pattern, content, re.DOTALL)
        
        room_assignments = {}
        
        # Map column positions to room names (based on typical MCM schedule layout)
        room_names = ["HH Auditorium", "HH Ballroom", "HH 002", "WH Auditorium", "WH 115"]
        
        for row in table_rows:
            # Find all \tableTalk entries in this row
            talks_in_row = re.findall(r'\\tableTalk\{([^}]*)\}\s*\{([^}]*)\}\s*\{([^}]*)\}', row)
            
            for i, (speaker, title, talk_id) in enumerate(talks_in_row):
                if talk_id.strip():
                    room_name = room_names[i] if i < len(room_names) else f"Room {i+1}"
                    room_assignments[talk_id.strip()] = {
                        'room': room_name,
                        'speaker': speaker.strip(),
                        'title': title.strip()
                    }
        
        # Update existing abstract data with room information
        for talk_id, room_info in room_assignments.items():
            if talk_id in st.session_state.abstracts_data:
                if 'schedule_info' not in st.session_state.abstracts_data[talk_id]:
                    st.session_state.abstracts_data[talk_id]['schedule_info'] = {}
                
                st.session_state.abstracts_data[talk_id]['schedule_info']['room'] = room_info['room']
                # Don't override time_slot if it exists from comprehensive data
                if not st.session_state.abstracts_data[talk_id].get('time_slot'):
                    st.session_state.abstracts_data[talk_id]['schedule_info']['time'] = 'TBD'
        
        print(f"Updated {len(room_assignments)} talks with room information")

    def extract_parameters_from_content(self, content: str, max_params: int) -> List[str]:
        """Extract parameters from LaTeX content using brace counting"""
        params = []
        i = 0
        
        while i < len(content) and len(params) < max_params:
            # Skip whitespace and comments
            while i < len(content) and (content[i].isspace() or content[i] == '%'):
                if content[i] == '%':
                    # Skip to end of line
                    while i < len(content) and content[i] != '\n':
                        i += 1
                i += 1
            
            if i >= len(content):
                break
                
            # Look for opening brace
            if content[i] == '{':
                brace_count = 1
                start = i + 1
                i += 1
                
                # Find matching closing brace
                while i < len(content) and brace_count > 0:
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                    i += 1
                
                if brace_count == 0:
                    param = content[start:i-1].strip()
                    params.append(param)
                else:
                    break
            else:
                i += 1
        
        return params

    def extract_abstract_from_content(self, content: str, params: List[str]) -> str:
        """Extract abstract text from content after parameters"""
        # Find where the abstract starts (after all the parameters)
        # Skip past all the {param} entries
        param_end_pos = 0
        brace_count = 0
        params_found = 0
        
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    # Starting a new parameter
                    pass
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Finished a parameter
                    params_found += 1
                    if params_found >= len(params):
                        param_end_pos = i + 1
                        break
        
        # Extract abstract text after parameters
        if param_end_pos < len(content):
            abstract_text = content[param_end_pos:].strip()
            # Clean the abstract text
            abstract_text = self.clean_abstract_content(abstract_text)
            return abstract_text
        
        return ""

    def extract_description_from_content(self, content: str, params: List[str]) -> str:
        """Extract session description from content after parameters"""
        return self.extract_abstract_from_content(content, params)

    def load_schedule_data(self):
        """Load schedule data from CSV files"""
        try:
            if self.schedule_path.exists():
                st.session_state.schedule_df = pd.read_csv(self.schedule_path)
                st.info(f"Loaded {len(st.session_state.schedule_df)} schedule items")
            else:
                # Create a basic schedule structure if file doesn't exist
                st.session_state.schedule_df = pd.DataFrame(columns=['Time', 'Session', 'Speaker', 'Title', 'Room'])
                st.warning("Schedule file not found, using empty schedule")
        except Exception as e:
            st.error(f"Error loading schedule data: {str(e)}")
            raise e
    
    def parse_abstract(self, content: str, file_id: str) -> Optional[Dict]:
        """Parse LaTeX abstract content"""
        try:
            print(f"Parsing {file_id}: content length = {len(content)}")
            # Extract talk information
            if '\\begin{talk}' in content:
                print(f"  -> Found talk in {file_id}")
                return self.parse_talk_abstract(content, file_id)
            elif '\\begin{session}' in content:
                print(f"  -> Found session in {file_id}")
                return self.parse_session_abstract(content, file_id)
            else:
                # Plenary talks
                print(f"  -> Treating as plenary: {file_id}")
                return self.parse_plenary_abstract(content, file_id)
        except Exception as e:
            print(f"Error parsing {file_id}: {e}")
            return None
    
    def parse_talk_abstract(self, content: str, file_id: str) -> Dict:
        """Parse contributed talk abstract with proper parameter extraction"""
        if IS_DEBUG_APP:
            print(f"DEBUG: Parsing {file_id}")
        
        # Find the \begin{talk} section and extract the 6 parameters
        talk_match = re.search(r'\\begin\{talk\}(.*?)(?=%Your abstract goes here|The analysis|Hit and run|\\end\{talk\})', content, re.DOTALL)
        
        if not talk_match:
            print(f"ERROR: Could not find talk environment in {file_id}")
            return {
                'type': 'talk',
                'id': file_id,
                'title': "Parse Error - No Talk Environment",
                'speaker': "Unknown Speaker",
                'affiliation': "",
                'email': "",
                'abstract': "",
                'content': content
            }
        
        talk_content = talk_match.group(1).strip()
        if IS_DEBUG_APP:
            print(f"DEBUG: Extracted talk content length: {len(talk_content)}")
        
        # Extract the 6 parameters in curly braces after \begin{talk}
        # Pattern: { content } with potential comments and whitespace
        params = []
        i = 0
        while i < len(talk_content) and len(params) < 6:
            # Skip whitespace and comments
            while i < len(talk_content) and (talk_content[i].isspace() or talk_content[i] == '%'):
                if talk_content[i] == '%':
                    # Skip to end of line
                    while i < len(talk_content) and talk_content[i] != '\n':
                        i += 1
                i += 1
            
            if i >= len(talk_content):
                break
                
            # Look for opening brace
            if talk_content[i] == '{':
                brace_count = 1
                start = i + 1
                i += 1
                
                # Find matching closing brace
                while i < len(talk_content) and brace_count > 0:
                    if talk_content[i] == '{':
                        brace_count += 1
                    elif talk_content[i] == '}':
                        brace_count -= 1
                    i += 1
                
                if brace_count == 0:
                    param = talk_content[start:i-1].strip()
                    params.append(param)
                    if IS_DEBUG_APP:
                        print(f"DEBUG: Found parameter {len(params)}: '{param}'")
                else:
                    if IS_DEBUG_APP:
                        print(f"ERROR: Unmatched braces in {file_id}")
                    break
            else:
                i += 1
        
        # Extract parameters with defaults
        title = params[0] if len(params) > 0 else "Unknown Title"
        speaker = params[1] if len(params) > 1 else "Unknown Speaker" 
        affiliation = params[2] if len(params) > 2 else ""
        email = params[3] if len(params) > 3 else ""
        coauthors = params[4] if len(params) > 4 else ""
        special_session = params[5] if len(params) > 5 else ""
        
        if IS_DEBUG_APP:
            print(f"DEBUG: Parsed {file_id} - Title: '{title}', Speaker: '{speaker}', Affiliation: '{affiliation}'")
        
        # Extract abstract text (everything after the parameters until \end{talk})
        abstract_text = ""
        abstract_start_patterns = [
            r'%Your abstract goes here[^\n]*\n',
            r'\n\s*The analysis of',
            r'\n\s*Hit and run',
            r'\}\s*\n\s*[A-Z]'  # Last } followed by text starting with capital letter
        ]
        
        abstract_end = content.find('\\end{talk}')
        if abstract_end != -1:
            # Find where abstract text starts
            for pattern in abstract_start_patterns:
                match = re.search(pattern, content)
                if match and match.end() < abstract_end:
                    abstract_text = content[match.end():abstract_end].strip()
                    break
            
            # Fallback: find text after the last parameter
            if not abstract_text and talk_match:
                remaining_content = content[talk_match.end():abstract_end].strip()
                # Remove leading comments and whitespace
                lines = remaining_content.split('\n')
                text_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('%'):
                        text_lines.append(line)
                abstract_text = '\n'.join(text_lines)
        
        return {
            'type': 'talk',
            'id': file_id,
            'title': title,
            'speaker': speaker,
            'affiliation': affiliation,
            'email': email,
            'coauthors': coauthors,
            'special_session': special_session,
            'abstract': abstract_text,
            'content': content
        }
    
    def parse_session_abstract(self, content: str, file_id: str) -> Dict:
        """Parse special session abstract with improved extraction"""
        if IS_DEBUG_APP:
            print(f"DEBUG: Parsing session {file_id}")
        
        # Extract session title from \begin{session} with 5 parameters
        session_match = re.search(r'\\begin\{session\}(.*?)(?=\([^)]*Monte Carlo|\n\n|\\end\{session\})', content, re.DOTALL)
        
        title = "Unknown Session"
        organizers = []
        
        if session_match:
            session_content = session_match.group(1).strip()
            if IS_DEBUG_APP:
                print(f"DEBUG: Found session content length: {len(session_content)}")
            
            # Extract the session title (first parameter in curly braces)
            title_match = re.search(r'\{([^}]+)\}', session_content)
            if title_match:
                title = title_match.group(1).strip()
                if IS_DEBUG_APP:
                    print(f"DEBUG: Extracted session title: '{title}'")
            
            # Extract organizer information
            organizer_pattern = r'\\organizer\{([^}]*)\}\s*\{([^}]*)\}\s*\{([^}]*)\}'
            organizer_matches = re.findall(organizer_pattern, content)
            
            for name, affil, email in organizer_matches:
                name = name.strip()
                affil = affil.strip()
                email = email.strip()
                if name:  # Only add if name is not empty
                    organizers.append({
                        'name': name,
                        'affiliation': affil,
                        'email': email
                    })
                    if IS_DEBUG_APP:
                        print(f"DEBUG: Found organizer: {name} ({affil})")
        
        # Extract session description (everything after \end{session} until \end{document})
        desc_start = content.find('\\end{session}')
        desc_end = content.find('\\end{document}')
        
        description = ""
        if desc_start != -1 and desc_end != -1:
            desc_content = content[desc_start + len('\\end{session}'):desc_end].strip()
            # Clean up the description
            lines = desc_content.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('%') and not line.startswith('\\begin{comment}'):
                    if '\\begin{comment}' in line:
                        break  # Stop at comment block
                    clean_lines.append(line)
            description = '\n'.join(clean_lines)
        
        if IS_DEBUG_APP: print(f"DEBUG: Session {file_id} - Title: '{title}', Organizers: {len(organizers)}, Description length: {len(description)}")
        
        return {
            'type': 'session',
            'id': file_id,
            'title': title,
            'organizers': organizers,
            'description': description,
            'content': content
        }
    
    def parse_plenary_abstract(self, content: str, file_id: str) -> Dict:
        """Parse plenary talk abstract"""
        # For plenary talks, try to extract basic information
        lines = content.split('\\n')
        title = "Plenary Talk"
        speaker = "Unknown Speaker"
        
        # Look for talk or plenary patterns
        for line in lines:
            if 'talk' in line.lower() and '{' in line:
                title_match = re.search(r'{([^}]*)}', line)
                if title_match:
                    title = title_match.group(1).strip()
            if 'speaker' in line.lower() and '{' in line:
                speaker_match = re.search(r'{([^}]*)}', line)
                if speaker_match:
                    speaker = speaker_match.group(1).strip()
        
        return {
            'type': 'plenary',
            'id': file_id,
            'title': title,
            'speaker': speaker,
            'affiliation': "",
            'email': "",
            'abstract': content,
            'content': content
        }
    
    def parse_talk_from_main_file(self, content: str, talk_id: str) -> Dict:
        """Parse talk content from main LaTeX files (contributed_talk_submissions_talks.tex, plenary_abstracts_talks.tex, special_session_abstracts_talks.tex)"""
        if IS_DEBUG_APP:
            print(f"DEBUG: Parsing talk from main file: {talk_id}")
        
        # These files have the complete structure with all parameters
        # Extract the parameters using the same logic as parse_talk_abstract
        talk_match = re.search(r'\\begin\{talk\}(.*?)(?=\\end\{talk\})', content, re.DOTALL)
        
        if not talk_match:
            print(f"ERROR: Could not find talk environment in {talk_id}")
            return None
        
        talk_content_inner = talk_match.group(1).strip()
        #print(f"DEBUG: Extracted talk content length: {len(talk_content_inner)}")
        
        # Extract the 9 parameters for main file format:
        # [1] title, [2] speaker, [3] affiliation, [4] email, [5] coauthors, 
        # [6] special session, [7] time slot, [8] talk id, [9] session id
        params = []
        i = 0
        while i < len(talk_content_inner) and len(params) < 9:
            # Skip whitespace and comments
            while i < len(talk_content_inner) and (talk_content_inner[i].isspace() or talk_content_inner[i] == '%'):
                if talk_content_inner[i] == '%':
                    # Skip to end of line
                    while i < len(talk_content_inner) and talk_content_inner[i] != '\n':
                        i += 1
                i += 1
            
            if i >= len(talk_content_inner):
                break
                
            # Look for opening brace
            if talk_content_inner[i] == '{':
                brace_count = 1
                start = i + 1
                i += 1
                
                # Find matching closing brace
                while i < len(talk_content_inner) and brace_count > 0:
                    if talk_content_inner[i] == '{':
                        brace_count += 1
                    elif talk_content_inner[i] == '}':
                        brace_count -= 1
                    i += 1
                
                if brace_count == 0:
                    param = talk_content_inner[start:i-1].strip()
                    params.append(param)
                    #print(f"DEBUG: Found parameter {len(params)}: '{param}'")
                else:
                    print(f"ERROR: Unmatched braces in {talk_id}")
                    break
            else:
                i += 1
        
        # Extract parameters with defaults
        title = params[0] if len(params) > 0 else "Unknown Title"
        speaker = params[1] if len(params) > 1 else "Unknown Speaker" 
        affiliation = params[2] if len(params) > 2 else ""
        email = params[3] if len(params) > 3 else ""
        coauthors = params[4] if len(params) > 4 else ""
        special_session = params[5] if len(params) > 5 else ""
        time_slot = params[6] if len(params) > 6 else ""
        session_id = params[8] if len(params) > 8 else ""
        
        #print(f"DEBUG: Parsed main file {talk_id} - Title: '{title}', Speaker: '{speaker}', Time: '{time_slot}'")
        
        # Extract abstract text (everything after the parameters)
        abstract_text = ""
        # Find the end of parameters and start of abstract content
        if talk_match:
            # Look for content after the last parameter
            remaining_content = talk_content_inner
            # Find where the abstract starts (after all the parameter braces)
            lines = remaining_content.split('\n')
            abstract_lines = []
            found_abstract_start = False
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                # If we find a line that doesn't start with { and isn't a comment, it's probably abstract content
                if not line.startswith('{') and not found_abstract_start:
                    found_abstract_start = True
                if found_abstract_start:
                    abstract_lines.append(line)
            
            abstract_text = '\n'.join(abstract_lines).strip()
        
        return {
            'type': 'talk',
            'id': talk_id,
            'title': title,
            'speaker': speaker,
            'affiliation': affiliation,
            'email': email,
            'coauthors': coauthors,
            'special_session': special_session,
            'time_slot': time_slot,
            'session_id': session_id,
            'abstract': abstract_text,
            'content': content
        }
    
    def parse_session_from_main_file(self, content: str, session_id: str) -> Dict:
        """Parse session content from special_session_submissions_talks.tex"""
        #print(f"DEBUG: Parsing session from main file: {session_id}")
        
        # Extract session parameters from the \begin{session}...\end{session} content
        session_match = re.search(r'\\begin\{session\}(.*?)\\end\{session\}', content, re.DOTALL)
        
        if not session_match:
            print(f"ERROR: Could not find session environment in {session_id}")
            return None
        
        session_content_inner = session_match.group(1).strip()
        #print(f"DEBUG: Extracted session content length: {len(session_content_inner)}")
        
        # Extract the 9 parameters for session format:
        # [1] title, [2] organizer1 name, [3] organizer1 affiliation, [4] organizer1 email
        # [5] organizer2 name, [6] organizer2 affiliation, [7] organizer2 email, [8] session id, [9] third organizer
        params = []
        i = 0
        while i < len(session_content_inner) and len(params) < 9:
            # Skip whitespace and comments
            while i < len(session_content_inner) and (session_content_inner[i].isspace() or session_content_inner[i] == '%'):
                if session_content_inner[i] == '%':
                    # Skip to end of line
                    while i < len(session_content_inner) and session_content_inner[i] != '\n':
                        i += 1
                i += 1
            
            if i >= len(session_content_inner):
                break
                
            # Look for opening brace
            if session_content_inner[i] == '{':
                brace_count = 1
                start = i + 1
                i += 1
                
                # Find matching closing brace
                while i < len(session_content_inner) and brace_count > 0:
                    if session_content_inner[i] == '{':
                        brace_count += 1
                    elif session_content_inner[i] == '}':
                        brace_count -= 1
                    i += 1
                
                if brace_count == 0:
                    param = session_content_inner[start:i-1].strip()
                    params.append(param)
                    #print(f"DEBUG: Found session parameter {len(params)}: '{param}'")
                else:
                    print(f"ERROR: Unmatched braces in session {session_id}")
                    break
            else:
                i += 1
        
        # Extract parameters with defaults
        title = params[0] if len(params) > 0 else "Unknown Session"
        organizer1_name = params[1] if len(params) > 1 else ""
        organizer1_affiliation = params[2] if len(params) > 2 else ""
        organizer1_email = params[3] if len(params) > 3 else ""
        organizer2_name = params[4] if len(params) > 4 else ""
        organizer2_affiliation = params[5] if len(params) > 5 else ""
        organizer2_email = params[6] if len(params) > 6 else ""
        third_organizer = params[8] if len(params) > 8 else ""
        
        # Build organizers list
        organizers = []
        if organizer1_name:
            organizers.append({
                'name': organizer1_name,
                'affiliation': organizer1_affiliation,
                'email': organizer1_email
            })
        if organizer2_name:
            organizers.append({
                'name': organizer2_name,
                'affiliation': organizer2_affiliation,
                'email': organizer2_email
            })
        
        # Extract third organizer if present
        if third_organizer and third_organizer.startswith('\\thirdorganizer{'):
            third_match = re.search(r'\\thirdorganizer\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}', third_organizer)
            if third_match:
                organizers.append({
                    'name': third_match.group(1).strip(),
                    'affiliation': third_match.group(2).strip(),
                    'email': third_match.group(3).strip()
                })
        
        #print(f"DEBUG: Parsed session {session_id} - Title: '{title}', Organizers: {len(organizers)}")
        
        # Extract session description (text after parameters)
        description = ""
        if session_match:
            # Look for content after the last parameter
            lines = session_content_inner.split('\n')
            desc_lines = []
            found_desc_start = False
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                # If we find a line that doesn't start with { and isn't a comment, it's probably description
                if not line.startswith('{') and not line.startswith('\\thirdorganizer') and not found_desc_start:
                    found_desc_start = True
                if found_desc_start:
                    desc_lines.append(line)
            
            description = '\n'.join(desc_lines).strip()
        
        return {
            'type': 'session',
            'id': session_id,
            'title': title,
            'organizers': organizers,
            'description': description,
            'content': content
        }
    
    def calculate_relevance_score(self, talk_data: Dict, interests: Dict) -> float:
        """Calculate how relevant a talk is to user interests using AI evaluation"""
        
        # Use AI evaluation if available
        if self.llm.available:
            ai_result = self.llm.evaluate_talk_relevance(talk_data, interests)
            
            # Store AI reasoning for display purposes
            if 'ai_reasoning' not in talk_data:
                talk_data['ai_reasoning'] = ai_result.get('reasoning', 'No reasoning provided')
            
            return ai_result.get('score', 0.0)
        
        # Fallback to basic algorithmic scoring if AI is unavailable
        return self._fallback_relevance_score(talk_data, interests)
    
    def calculate_relevance_scores_batch(self, talks: List[Dict], interests: Dict) -> List[Dict]:
        """Calculate relevance scores for multiple talks efficiently"""
        if not talks:
            return []
        
        # Use AI batch evaluation if available
        if self.llm.available:
            return self.llm.evaluate_talks_batch(talks, interests)
        
        # Fallback to individual scoring
        results = []
        for talk in talks:
            score = self._fallback_relevance_score(talk, interests)
            results.append({
                'talk': talk,
                'score': score,
                'reasoning': 'Basic algorithmic scoring (AI unavailable)'
            })
        return results
    
    def _fallback_relevance_score(self, talk_data: Dict, interests: Dict) -> float:
        """Fallback algorithmic scoring when AI is unavailable"""
        score = 0.0
        
        # Safely get text fields
        title = talk_data.get('title', talk_data.get('Title', ''))
        abstract = talk_data.get('abstract', talk_data.get('Abstract', ''))
        description = talk_data.get('description', talk_data.get('Description', ''))
        speaker = talk_data.get('speaker', talk_data.get('Speaker', ''))
        
        # Handle organizers (might be list or string)
        organizers_text = ''
        organizers = talk_data.get('organizers', [])
        if isinstance(organizers, list):
            organizers_text = ' '.join([org.get('name', '') if isinstance(org, dict) else str(org) for org in organizers])
        elif isinstance(organizers, str):
            organizers_text = organizers
        
        # Combine all text for searching
        search_text = ' '.join([
            title,
            abstract,
            description,
            organizers_text,
            speaker
        ]).lower()
        
        # Skip if no meaningful text content
        if not search_text.strip():
            return 0.0
        
        # Basic keyword matching (50% of score)
        keyword_score = 0
        for keyword in interests.get('keywords', []):
            if keyword.lower() in search_text:
                keyword_score += 1
        if interests.get('keywords'):
            keyword_score = (keyword_score / len(interests['keywords'])) * 50
        
        # Research area matching (30% of score)
        area_score = 0
        for area in interests.get('areas', []):
            if area.lower() in search_text:
                area_score += 1
        if interests.get('areas'):
            area_score = (area_score / len(interests['areas'])) * 30
        
        # Content type relevance (20% of score)
        type_score = 0
        talk_type = talk_data.get('type', 'contributed')
        
        if talk_type == 'plenary':
            type_score = 20
        elif talk_type == 'session':
            type_score = 15
        else:
            type_score = 10
        
        score = keyword_score + area_score + type_score
        return min(score, 100.0)
    
    def get_schedule_info(self, talk_id: str, talk_data: Dict) -> Optional[Dict]:
        """Get schedule information for a talk"""
        for schedule_item in st.session_state.schedule_data:
            # Try to match by various criteria
            title_match = False
            if 'SessionTitle' in schedule_item:
                session_title = schedule_item['SessionTitle'].lower()
                talk_title = talk_data.get('title', '').lower()
                
                # Check for partial matches
                if talk_title and any(word in session_title for word in talk_title.split() if len(word) > 3):
                    title_match = True
                elif talk_data.get('type') == 'session' and talk_title in session_title:
                    title_match = True
            
            if title_match:
                return {
                    'time': schedule_item.get('SessionTime', ''),
                    'room': schedule_item.get('Room', ''),
                    'chair': schedule_item.get('Chair', '')
                }
        
        # Default schedule info if not found
        return {
            'time': 'TBD',
            'room': 'TBD',
            'chair': 'TBD'
        }
    
    def get_system_instructions(self) -> str:
        """Get system instructions for AI model (not shown to user)"""
        return """
        SYSTEM INSTRUCTIONS FOR TALK SELECTION:
        - When multiple parallel sessions cover similar topics, select the talk that appears most comprehensive or directly aligns with the user's stated interests
        - Note that each special session (e.g., 'Hardware or Software for (Quasi-)Monte Carlo Algorithms, Part I') contains 3-4 talks
        - Usually there are parallel special sessions and technical sessions - select just one from parallel options
        - Each plenary talk comes before parallel sessions in each morning or afternoon.
        - The system will automatically enforce maximum 2 plenary sessions, 2 special or technical sessions (i.e., 10 talks per day) and maximum 45 talks total for the conference.
        - Prioritize quality and relevance over quantity
        - Include coffee/lunch breaks and banquets.

        SYSTEM INSTRUCTIONS FOR TALK ABSTRACT FORMAT:
        - Each select talk abstract should start with meta data of the talk
        - Followed by the abstract text
        - After the abstract of each talk, create a summary list of 3-7 key takeaways or insights from the talk in bullet points.
        """
    
    def combine_preferences_with_system_instructions(self, user_preferences: str) -> str:
        """Combine user preferences with system instructions for AI model"""
        system_instructions = self.get_system_instructions()
        
        if user_preferences.strip():
            return f"{user_preferences.strip()}\n\n{system_instructions.strip()}"
        else:
            return system_instructions.strip()
    
    def search_talks(self, keywords: List[str], areas: List[str], experience: str, preferences: str):
        """Search for relevant talks based on user interests"""
        # Combine user preferences with system instructions for AI
        combined_preferences = self.combine_preferences_with_system_instructions(preferences)
        
        user_interests = {
            'keywords': keywords,
            'areas': areas,
            'preferences': combined_preferences,  # Use combined preferences for AI
            'experience': experience,
            'user_preferences_only': preferences  # Keep original user preferences for display
        }
        
        # Check if interests have changed significantly - if so, clear cache
        if hasattr(st.session_state, 'last_user_interests'):
            last_interests = st.session_state.last_user_interests
            interests_changed = (
                set(keywords) != set(last_interests.get('keywords', [])) or
                set(areas) != set(last_interests.get('areas', [])) or
                preferences != last_interests.get('user_preferences_only', '') or
                experience != last_interests.get('experience', '')
            )
            if interests_changed:
                self.llm.clear_evaluation_cache()
                st.info("🔄 Research interests changed - cleared AI cache for fresh analysis")
        
        st.session_state.user_interests = user_interests
        st.session_state.last_user_interests = user_interests.copy()
        
        # Show preferences being applied (brief confirmation only)
        if preferences.strip() and self.llm.available:
            # Just show a brief confirmation since the spinner already shows the main analysis message
            pass  # The spinner message above already covers this
        elif preferences.strip() and not self.llm.available:
            st.warning("🤖 AI service unavailable - using basic preference matching. Start Ollama for intelligent preference analysis.")
        
        # Ensure we have both schedule data and abstracts
        if not hasattr(st.session_state, 'schedule_data') or not st.session_state.schedule_data:
            st.error("Schedule data not loaded. Please check data files.")
            return []
        
        # Calculate relevance scores for both schedule items and abstracts
        relevant_talks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Combine schedule data with abstract data, prioritizing abstracts
        all_talks = []
        
        # Add abstract data first (higher priority for detailed content)
        for talk_id, abstract_data in st.session_state.abstracts_data.items():
            all_talks.append({
                'source': 'abstract',
                'data': abstract_data,
                'id': talk_id
            })
        
        # Add schedule items (sessions, plenaries, etc.) - lower priority
        for schedule_item in st.session_state.schedule_data:
            all_talks.append({
                'source': 'schedule',
                'data': schedule_item
            })
        
        total_talks = len(all_talks)
        
        if total_talks == 0:
            st.error("No talks found. Please check that data files are properly loaded.")
            return []
        
        # Pre-filter talks based on basic criteria to reduce AI processing load
        filtered_talks = []
        basic_filter_keywords = [kw.lower() for kw in user_interests.get('keywords', [])]
        
        for talk_item in all_talks:
            unified_talk = self.create_unified_talk_object(talk_item)
            
            # Quick pre-filter: if no keywords match at all, skip expensive AI analysis
            if basic_filter_keywords and self.llm.available:
                talk_text = ' '.join([
                    unified_talk.get('title', ''),
                    unified_talk.get('abstract', ''),
                    unified_talk.get('speaker', '')
                ]).lower()
                
                # Only include if at least one keyword appears or it's a high-value talk type
                has_keyword_match = any(kw in talk_text for kw in basic_filter_keywords)
                is_high_value = unified_talk.get('type') in ['plenary', 'session']
                
                if not (has_keyword_match or is_high_value):
                    continue
            
            filtered_talks.append(unified_talk)
        
        #print(f"DEBUG: Pre-filtered {len(all_talks)} talks down to {len(filtered_talks)} for detailed analysis")
        
        # Calculate optimal batch size using intelligent algorithm
        batch_size = self.llm.calculate_optimal_batch_size(filtered_talks)
        
        # Show batch strategy info
        if self.llm.available:
            avg_content_length = sum(len(talk.get('abstract', '') + talk.get('title', '')) 
                                   for talk in filtered_talks[:min(10, len(filtered_talks))]) / min(10, len(filtered_talks)) if filtered_talks else 0
            if IS_DEBUG_APP:
                print(f"DEBUG: Optimal batch_size={batch_size} for {len(filtered_talks)} talks (avg_length={avg_content_length:.0f} chars)")
        
        relevant_talks = []
        
        for i in range(0, len(filtered_talks), batch_size):
            batch = filtered_talks[i:i + batch_size]
            batch_start_time = time.time()
            
            if self.llm.available:
                # Use batch processing for AI
                try:
                    batch_results = self.calculate_relevance_scores_batch(batch, user_interests)
                    
                    # Monitor performance and adjust batch size if needed
                    batch_time = time.time() - batch_start_time
                    config = self.llm.batch_config
                    
                    if batch_time > config['timeout_threshold']:  # Batch taking too long
                        new_batch_size = max(config['min_batch_size'], int(batch_size * 0.7))
                        if new_batch_size != batch_size:
                            if IS_DEBUG_APP:
                                print(f"DEBUG: Batch took {batch_time:.1f}s, reducing batch_size from {batch_size} to {new_batch_size}")
                            batch_size = new_batch_size
                    elif batch_time < config['fast_threshold'] and batch_size < config['max_batch_size']:
                        new_batch_size = min(config['max_batch_size'], batch_size + 2)
                        if new_batch_size != batch_size:
                            if IS_DEBUG_APP:
                                print(f"DEBUG: Batch was fast ({batch_time:.1f}s), increasing batch_size to {new_batch_size}")
                            batch_size = new_batch_size
                    
                    for result in batch_results:
                        talk = result['talk']
                        score = result['score']
                        
                        if score > 0:
                            talk['relevance_score'] = score
                            talk['ai_reasoning'] = result.get('reasoning', 'AI analysis completed')
                            relevant_talks.append(talk)
                            
                except Exception as e:
                    #print(f"DEBUG: Batch processing failed, falling back to individual processing: {e}")
                    # Fallback to individual processing for this batch
                    for talk in batch:
                        score = self.calculate_relevance_score(talk, user_interests)
                        if score > 0:
                            talk['relevance_score'] = score
                            relevant_talks.append(talk)
            else:
                # Use individual processing for fallback
                for talk in batch:
                    score = self.calculate_relevance_score(talk, user_interests)
                    if score > 0:
                        talk['relevance_score'] = score
                        relevant_talks.append(talk)
            
            # Update progress
            progress = min((i + batch_size) / len(filtered_talks), 1.0)
            progress_bar.progress(progress)
            
            if self.llm.available:
                status_text.text(f"🚀 AI batch processing {min(i + batch_size, len(filtered_talks))}/{len(filtered_talks)} talks...")
            else:
                status_text.text(f"⚙️ Processing {min(i + batch_size, len(filtered_talks))}/{len(filtered_talks)} talks (basic mode)...")
        
        # Final status update
        if self.llm.available:
            speedup_info = f"Pre-filtered {total_talks}→{len(filtered_talks)} talks, processed in batches of {batch_size}"
            st.info(f"🚀 **Performance optimized!** {speedup_info}")
        
        # Sort by relevance
        relevant_talks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        st.session_state.current_talks = relevant_talks
        
        progress_bar.empty()
        status_text.empty()
        
        # Show preferences impact summary
        if preferences.strip() and relevant_talks:
            excluded_count = sum(1 for talk in relevant_talks if talk['relevance_score'] == 0)
            high_score_count = sum(1 for talk in relevant_talks if talk['relevance_score'] > 80)
            
            if self.llm.available:
                st.success(f"🤖 **AI preferences analysis complete!** ")
                
                # Show some AI reasoning examples for top talks
                if high_score_count > 0:
                    with st.expander("🧠 AI Reasoning Examples", expanded=False):
                        for i, talk in enumerate(relevant_talks[:3]):  # Show top 3
                            if talk.get('ai_reasoning'):
                                st.write(f"**{talk.get('title', 'Unknown')}** (Score: {talk.get('relevance_score', 0):.0f})")
                                st.write(f"*AI Analysis:* {talk['ai_reasoning']}")
                                st.write("---")
            else:
                if excluded_count > 0 or high_score_count > 0:
                    st.success(f"⚙️ **Basic preferences applied!** "
                              f"Excluded {excluded_count} talks, boosted {high_score_count} highly relevant talks.")
                    st.info("💡 **Tip:** Start Ollama AI service for intelligent preference analysis and detailed reasoning.")
        
        return relevant_talks
    
    def create_unified_talk_object(self, talk_item: Dict) -> Dict:
        """Create a unified talk object from schedule or abstract data"""
        source = talk_item['source']
        data = talk_item['data']
        
        if source == 'schedule':
            # Handle schedule data
            return {
                'source': 'schedule',
                'title': data.get('SessionTitle', ''),
                'speaker': data.get('Chair', ''),  # Chair as speaker for sessions
                'time': data.get('SessionTime', ''),
                'room': data.get('Room', ''),
                'abstract': data.get('SessionTitle', ''),  # Use title as searchable text
                'type': 'session',
                'schedule_info': {
                    'time': data.get('SessionTime', ''),
                    'room': data.get('Room', ''),
                    'chair': data.get('Chair', '')
                }
            }
        elif source == 'abstract':
            # Handle abstract data
            title = data.get('title', '')
            speaker = data.get('speaker', '')
            
            # Prioritize time slot from comprehensive LaTeX data if available
            schedule_info = {}
            if data.get('time_slot'):
                # Use comprehensive time slot from main LaTeX files (most reliable)
                # Also use room information if available from consolidated parsing
                room = 'TBD'
                #print(f"***DEBUG {data.keys() = }")
                if data.get('schedule_info', {}).get('room'):
                    room = data['schedule_info']['room']
                
                schedule_info = {
                    'time': data.get('time_slot'),
                    'room': room,
                    'chair': 'TBD'  # Chair info not typically in LaTeX files
                }
                #print(f"DEBUG: Using comprehensive time slot for {talk_item.get('id', '')}: {data.get('time_slot')}, Room: {room}")
            else:
                # Fallback to CSV schedule matching for older data
                print(f"\n***DEBUG: {talk_item = }\n")
                schedule_info = self.find_schedule_for_abstract(talk_item.get('id', ''), data)
                #print(f"DEBUG: Using CSV schedule fallback for {talk_item.get('id', '')}: {schedule_info.get('time', 'TBD')}")
            
            return {
                'source': 'abstract',
                'id': talk_item.get('id', ''),  # Include the ID so we can find the LaTeX file
                'title': title,
                'speaker': speaker,
                'affiliation': data.get('affiliation', ''),
                'email': data.get('email', ''),
                'abstract': data.get('abstract', ''),
                'type': data.get('type', 'contributed'),
                'time_slot': data.get('time_slot', ''),  # Include comprehensive time slot
                'session_id': data.get('session_id', ''),  # Include session ID
                'schedule_info': schedule_info
            }
        
        return {}
    
    def find_schedule_for_abstract(self, talk_id: str, abstract_data: Dict) -> Dict:
        """Find schedule information for an abstract"""
        # Try to match abstract with schedule data
        title = abstract_data.get('title', '').lower()
        speaker = abstract_data.get('speaker', '').lower()
        
        for schedule_item in st.session_state.schedule_data:
            session_title = schedule_item.get('SessionTitle', '').lower()
            chair = schedule_item.get('Chair', '').lower()
            
            # Match by title keywords or speaker name
            if title and any(word in session_title for word in title.split() if len(word) > 3):
                return {
                    'time': schedule_item.get('SessionTime', 'TBD'),
                    'room': schedule_item.get('Room', 'TBD'),
                    'chair': schedule_item.get('Chair', 'TBD')
                }
            elif speaker and speaker in chair:
                return {
                    'time': schedule_item.get('SessionTime', 'TBD'),
                    'room': schedule_item.get('Room', 'TBD'),
                    'chair': schedule_item.get('Chair', 'TBD')
                }
        
        # Default if no match found
        return {
            'time': 'TBD',
            'room': 'TBD',
            'chair': 'TBD'
        }
    
    def check_conflicts(self, selected_talks: List[Dict]) -> Dict:
        """Check for scheduling conflicts"""
        conflicts = {}
        time_slots = {}
        
        for talk in selected_talks:
            schedule_info = talk.get('schedule_info', {})
            time = schedule_info.get('time', 'TBD')
            
            if time != 'TBD':
                if time in time_slots:
                    time_slots[time].append(talk)
                else:
                    time_slots[time] = [talk]
        
        # Find conflicts
        for time, talks_list in time_slots.items():
            if len(talks_list) > 1:
                conflicts[time] = talks_list
        
        return conflicts
    
    def select_talks_without_conflicts(self, talks_with_scores: List[Dict]) -> List[Dict]:
        """Select talks without scheduling conflicts using greedy algorithm with daily and total limits
        
        Enforces hard constraints:
        - Maximum 10 talks per day
        - Maximum 45 talks total for the conference
        - No scheduling conflicts (same time slot)
        
        Note: Preferences-based exclusions are now handled during relevance scoring,
        so talks that should be excluded based on user preferences will have score=0
        and won't make it to this selection phase.
        """
        selected_talks = []
        used_time_slots = set()
        daily_counts = {}  # Track talks per day
        
        # Define day mapping for time slots
        day_keywords = {
            'monday': 'Monday',
            'mon': 'Monday',
            'tuesday': 'Tuesday', 
            'tue': 'Tuesday',
            'wednesday': 'Wednesday',
            'wed': 'Wednesday',
            'thursday': 'Thursday',
            'thu': 'Thursday',
            'friday': 'Friday',
            'fri': 'Friday'
        }
        
        # Initialize daily counts
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            daily_counts[day] = 0
        
        def extract_day_from_time(time_slot: str) -> str:
            """Extract day from time slot string"""
            if not time_slot:
                return 'Unknown'
            
            time_lower = time_slot.lower()
            for keyword, day in day_keywords.items():
                if keyword in time_lower:
                    return day
            return 'Unknown'
        
        def normalize_time_slot(time_slot: str) -> str:
            """Normalize time slot for conflict detection"""
            if not time_slot or time_slot == 'TBD':
                return ''
            
            # Extract core time information (day + time range)
            # Examples: "Mon, Jul 28 09:00–10:00" -> "Mon 09:00–10:00"
            #           "Fri, Aug 1 09:30–10:00" -> "Fri 09:30–10:00"
            
            normalized = time_slot.strip()
            
            # Extract day pattern
            day_match = re.search(r'(Mon|Tue|Wed|Thu|Fri)', normalized)
            if not day_match:
                return normalized
            
            day = day_match.group(1)
            
            # Extract time range pattern (HH:MM–HH:MM)
            time_match = re.search(r'(\d{1,2}:\d{2}[–-]\d{1,2}:\d{2})', normalized)
            if not time_match:
                # Try to find start time at least
                start_time_match = re.search(r'(\d{1,2}:\d{2})', normalized)
                if start_time_match:
                    return f"{day} {start_time_match.group(1)}"
                return f"{day}"
            
            time_range = time_match.group(1)
            return f"{day} {time_range}"
        
        # Sort by relevance score (highest first)
        sorted_talks = sorted(talks_with_scores, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        for item in sorted_talks:
            talk = item['talk']
            time_slot = item.get('time', '')
            
            # Extract day from time slot
            day = extract_day_from_time(time_slot)
            
            # Normalize time slot for conflict detection
            normalized_time = normalize_time_slot(time_slot)
            
            # Skip if time slot is already used (conflict avoidance)
            if normalized_time and normalized_time in used_time_slots:
                if IS_DEBUG_APP:
                    print(f"DEBUG: Skipping conflicting talk: {talk.get('title', '')[:50]}... at {normalized_time}")
                continue
            
            # Skip if daily limit reached (10 talks per day)
            if day != 'Unknown' and daily_counts[day] >= 10:
                if IS_DEBUG_APP:
                    print(f"DEBUG: Skipping talk due to daily limit on {day}: {talk.get('title', '')[:50]}...")
                continue
            
            # Skip if total limit reached (45 talks total)
            if len(selected_talks) >= 45:
                if IS_DEBUG_APP:
                    print(f"DEBUG: Reached total limit of 45 talks")
                break
            
            # Add talk to selection
            selected_talks.append(item)
            if normalized_time:
                used_time_slots.add(normalized_time)
                if IS_DEBUG_APP:
                    print(f"DEBUG: Added talk at {normalized_time}: {talk.get('title', '')[:50]}...")
            
            # Update daily count
            if day != 'Unknown':
                daily_counts[day] += 1
        
        # Log the final distribution
        if IS_DEBUG_APP:
            print(f"DEBUG: Final talk distribution by day: {daily_counts}")
            print(f"DEBUG: Total talks selected: {len(selected_talks)}")
        
        return selected_talks
    
    def generate_pdf(self, selected_talks: List[Dict], filename: str, include_abstracts: bool, 
                    include_schedule: bool, include_conflicts: bool):
        """Generate the personalized PDF program book"""
        try:
            # Generate LaTeX content
            latex_content = self.create_latex_document(selected_talks, include_abstracts, 
                                                     include_schedule, include_conflicts)
            
            # Save LaTeX source to "out" directory
            out_dir = self.base_path.parent / "out"
            out_dir.mkdir(exist_ok=True)
            tex_filename = filename.replace('.pdf', '.tex')
            latex_source_file = out_dir / tex_filename
            
            # Write LaTeX source to out directory
            with open(latex_source_file, 'w', encoding='utf-8') as f:
                safe_latex_content = self.safe_text_encode(latex_content)
                f.write(safe_latex_content)
            
            # Write to temporary file for compilation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as tex_file:
                safe_latex_content = self.safe_text_encode(latex_content)
                tex_file.write(safe_latex_content)
                tex_file_path = tex_file.name
            
            output_file = self.output_path / filename
            
            # Compile to PDF
            self.compile_latex_to_pdf(tex_file_path, output_file)
            
            # Clean up temporary file only if successful
            os.unlink(tex_file_path)
            
            return output_file
            
        except Exception as e:
            # Save the problematic LaTeX file for debugging in both locations
            if 'tex_file_path' in locals():
                # Save to output directory
                debug_tex_file = self.output_path / f"debug_{filename.replace('.pdf', '.tex')}"
                # Also save to out directory
                out_dir = self.base_path.parent / "out"
                out_dir.mkdir(exist_ok=True)
                debug_out_file = out_dir / f"debug_{filename.replace('.pdf', '.tex')}"
                try:
                    import shutil
                    shutil.copy2(tex_file_path, debug_tex_file)
                    shutil.copy2(tex_file_path, debug_out_file)
                    error_msg = f"Failed to generate PDF: {str(e)}\nProblematic LaTeX file saved as: {debug_tex_file} and {debug_out_file}"
                except:
                    error_msg = f"Failed to generate PDF: {str(e)}"
            else:
                error_msg = f"Failed to generate PDF: {str(e)}"
            raise Exception(error_msg)
    
    def generate_ai_enhanced_pdf(self, selected_talks: List[Dict], filename: str, 
                               include_abstracts: bool, include_schedule: bool, 
                               include_conflicts: bool, ai_data: Dict):
        """Generate AI  personalized PDF program book"""
        try:
            # Generate LaTeX content with AI enhancements
            latex_content = self.create_ai_enhanced_latex_document(
                selected_talks, include_abstracts, include_schedule, 
                include_conflicts, ai_data
            )
            
            # Save LaTeX source to "out" directory
            out_dir = self.base_path.parent / "out"
            out_dir.mkdir(exist_ok=True)
            tex_filename = filename.replace('.pdf', '.tex')
            latex_source_file = out_dir / tex_filename
            
            # Write LaTeX source to out directory
            with open(latex_source_file, 'w', encoding='utf-8') as f:
                safe_latex_content = self.safe_text_encode(latex_content)
                f.write(safe_latex_content)
            
            # Write to temporary file for compilation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as tex_file:
                safe_latex_content = self.safe_text_encode(latex_content)
                tex_file.write(safe_latex_content)
                tex_file_path = tex_file.name
            
            output_file = self.output_path / filename
            
            # Compile to PDF
            self.compile_latex_to_pdf(tex_file_path, output_file)
            
            # Clean up temporary file
            os.unlink(tex_file_path)
            
            return output_file
            
        except Exception as e:
            raise Exception(f"Failed to generate AI  PDF: {str(e)}")
    
    def create_ai_enhanced_latex_document(self, selected_talks: List[Dict], 
                                        include_abstracts: bool, include_schedule: bool, 
                                        include_conflicts: bool, ai_data: Dict) -> str:
        """Create AI  LaTeX document content"""
        doc_parts = []
        
        # Enhanced document header with AI styling
        doc_parts.append(r"""
\documentclass[12pt,letterpaper,oneside]{book}

\usepackage{needspace}
\usepackage[dvipsnames]{xcolor}
\usepackage{xurl}
\usepackage[colorlinks=true,linkcolor=NavyBlue,citecolor=ForestGreen,urlcolor=Magenta]{hyperref}
\hypersetup{breaklinks=true}
\usepackage{bm}
\usepackage{verbatim}
\usepackage{tabularx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathrsfs}
\usepackage{enumitem}
\usepackage{fontawesome5}
\usepackage{tcolorbox}
\setlist[itemize]{itemsep=2pt, parsep=0pt}

% AI  styling
\definecolor{AIBlue}{RGB}{0,123,255}
\definecolor{AIGreen}{RGB}{40,167,69}
\definecolor{AIGray}{RGB}{108,117,125}

\newtcolorbox{aibox}[1][]{
    colback=blue!5!white,
    colframe=AIBlue,
    title={\faRobot\ AI Analysis},
    #1
}

\usepackage{fancyhdr}
\usepackage{datetime}

\newdateformat{myformat}{\twodigit{\THEDAY}~\monthname[\THEMONTH]~\THEYEAR}

\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.4pt}

\fancyfoot[C]{%
    \makebox[\textwidth]{%
        \myformat\today\ \currenttime\ \textcolor{AIBlue}{{\faRobot\ AI}} \textcolor{AIGray}{\footnotesize (""" + f"{self.llm.model_name if self.llm.available else 'N/A'}" + r""")}
        \hfill
        \thepage
    }
}

\makeatletter
\renewcommand{\footrule}{%
    \hrule width \textwidth height \footrulewidth \vskip2pt
}
\makeatother

\fancypagestyle{plain}{%
    \fancyhf{}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0.4pt}
\fancyfoot[C]{%
    \makebox[\textwidth]{%
        \myformat\today\ \currenttime\ \textcolor{AIBlue}{{\faRobot\ AI}} \textcolor{AIGray}{\footnotesize (""" + f"{self.llm.model_name if self.llm.available else 'N/A'}" + r""")}
        \hfill
        \thepage
    }
}
}

\pagestyle{fancy}

\usepackage[top=1in,bottom=1in,left=1in,right=1in,headheight=0pt,headsep=0pt,footskip=30pt]{geometry}

\begin{document}
""")
        
        # AI  title page
        doc_parts.append(r"""
\begin{titlepage}
\centering
\vspace*{1.5cm}

{\Huge\bfseries\textcolor{AIBlue}{\faRobot} MCM 2025}\\[0.3cm]
{\LARGE AI  Program Book}\\[1.5cm]

{\Large The Fifteenth International Conference on\\
Monte Carlo and Applications}\\[1cm]

{\large Illinois Institute of Technology\\
Chicago, Illinois, USA\\[1cm]
July 28 -- August 1, 2025}\\[1.5cm]

\begin{aibox}[width=0.8\textwidth]
This personalized program was intelligently curated using advanced AI analysis of your research interests, ensuring optimal relevance and conflict-free scheduling.
\end{aibox}

\vspace{1cm}
{\large Generated on \myformat\today}\\[0.5cm]
{\footnotesize AI Model: \textcolor{AIGray}{\texttt{""" + f"{self.llm.model_name if self.llm.available else 'N/A'}" + r"""}}}\\[1cm]

""")
        
        # Add user interests summary with AI insights
        if st.session_state.user_interests:
            interests = st.session_state.user_interests
            doc_parts.append(r"\begin{flushleft}")
            doc_parts.append(r"\textbf{\textcolor{AIBlue}{\faUser} Your Research Profile:}\\[0.5cm]")
            
            if interests.get('keywords'):
                keywords_str = ', '.join(interests['keywords'])
                doc_parts.append(f"\\textbf{{Keywords:}} {self.escape_latex(keywords_str)}\\\\[0.3cm]")
            
            if interests.get('areas'):
                areas_str = ', '.join(interests['areas'])
                doc_parts.append(f"\\textbf{{Research Areas:}} {self.escape_latex(areas_str)}\\\\[0.3cm]")
            
            doc_parts.append(f"\\textbf{{\\textcolor{{AIGreen}}{{AI-Selected Talks:}}}} {len(selected_talks)}\\\\")
            doc_parts.append(r"\end{flushleft}")
        
        doc_parts.append(r"\end{titlepage}")
        
        # AI Program Summary section
        if ai_data.get('include_ai_summary') and ai_data.get('program_summary'):
            doc_parts.append(r"""
\chapter{\textcolor{AIBlue}{\faRobot} AI Program Analysis}

\begin{aibox}
""")
            doc_parts.append(self.escape_latex(ai_data['program_summary']))
            doc_parts.append(r"""
\end{aibox}

\newpage
""")
        
        # Table of contents
        doc_parts.append(r"""
\tableofcontents
\newpage
""")
        
        # Schedule section with AI enhancements
        if include_schedule:
            doc_parts.append(self.create_ai_enhanced_schedule_section(selected_talks, ai_data))
        
        # Conflicts warning
        if include_conflicts:
            conflicts = self.check_conflicts(selected_talks)
            if conflicts:
                doc_parts.append(self.create_conflicts_section(conflicts))
        
        # AI  abstracts section
        if include_abstracts:
            doc_parts.append(self.create_ai_enhanced_abstracts_section(selected_talks, ai_data))
        
        doc_parts.append(r"\end{document}")
        
        return ''.join(doc_parts)
    
    def create_ai_enhanced_schedule_section(self, selected_talks: List[Dict], ai_data: Dict) -> str:
        """Create AI  schedule section"""
        content = [r"""
\chapter{\textcolor{AIBlue}{\faCalendarAlt} Your AI-Curated Schedule}

This section contains your personalized conference schedule, intelligently organized to maximize learning and minimize conflicts.

"""]
        
        # Group by day
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        schedule_by_day = {day: [] for day in days}
        schedule_by_day['Unknown'] = []  # Add Unknown day for unmatched talks
        
        for talk in selected_talks:
            schedule_info = talk.get('schedule_info', {})
            time_str = schedule_info.get('time', 'TBD')
            
            day = 'Unknown'
            for d in days:
                if d[:3] in time_str:
                    day = d
                    break
            
            schedule_by_day[day].append((talk, schedule_info))
        
        # Process known days first, then Unknown
        all_days = days + ['Unknown']
        for day in all_days:
            if schedule_by_day[day]:
                if day == 'Unknown':
                    content.append(f"\\section{{TBD / Unscheduled Sessions}}\n\n")
                else:
                    content.append(f"\\section{{{day}, July {28 + days.index(day) if days.index(day) < 4 else 1} {'2025' if days.index(day) < 4 else 'August 2025'}}}\n\n")
                
                # Sort by time
                schedule_by_day[day].sort(key=lambda x: x[1].get('time', ''))
                
                content.append(r"\begin{itemize}" + "\n")
                for talk, schedule_info in schedule_by_day[day]:
                    time_str = schedule_info.get('time', 'TBD')
                    room = schedule_info.get('room', 'TBD')
                    title = self.escape_latex(talk.get('title', ''))
                    speaker = self.escape_latex(talk.get('speaker', ''))
                    
                    # Clean time string - remove day/date, keep only time
                    clean_time = time_str
                    if time_str and time_str != 'TBD':
                        # Remove day and date patterns like "Mon, Jul 28 " or "Monday, Jul 28 "
                        clean_time = re.sub(r'^(Mon|Tue|Wed|Thu|Fri|Monday|Tuesday|Wednesday|Thursday|Friday),?\s*(Jul|Aug)\s*\d+\s*', '', time_str)
                        # Clean up any remaining formatting
                        clean_time = clean_time.strip()
                    
                    # Format time and room information
                    if clean_time != 'TBD' and room != 'TBD':
                        location_info = f"{clean_time}, {room}"
                    elif clean_time != 'TBD':
                        location_info = f"{clean_time}, TBD"
                    elif room != 'TBD':
                        location_info = f"TBD, {room}"
                    else:
                        location_info = "TBD"
                    
                    # Single line format: time/room, title, speaker
                    content.append(f"\\item {location_info}, {title}, {speaker}")
                    
                    # Add AI insights if available
                    if ai_data.get('include_ai_insights') and talk.get('llm_level'):
                        content.append(f", \\textcolor{{AIGray}}{{\\small \\faRobot\\ {talk.get('llm_level')} level}}")
                    
                    content.append("\n\n")
                
                content.append(r"\end{itemize}" + "\n\n")
        
        return ''.join(content)
    
    def create_ai_enhanced_abstracts_section(self, selected_talks: List[Dict], ai_data: Dict) -> str:
        """Create AI enhanced abstracts section using original LaTeX files"""
        content = [r"""
\chapter{\textcolor{AIBlue}{\faFileAlt} AI Enhanced Talk Abstracts}

This section contains detailed information about your selected talks, enhanced with AI analysis and using original LaTeX files for mathematical notation.

"""]
        
        # Sort talks by relevance score
        sorted_talks = sorted(selected_talks, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Set the path to the abstracts directory
        abstracts_base_path = "/Users/terrya/Documents/ProgramData/MCM-2025-Program/preprocess/input/abstracts"
        
        for i, talk in enumerate(sorted_talks):
            try:
                # Get the talk ID to find the corresponding .tex file
                talk_id = talk.get('id', '')
                
                # Add debug info
                content.append(f"% DEBUG: AI Talk {i+1}: ID = {talk_id}\n")
                content.append(f"% DEBUG: Relevance Score: {talk.get('relevance_score', 0):.1f}%\n")
                
                # Add AI insights box first if available
                if (ai_data.get('include_ai_insights') and 
                    any(key in talk for key in ['llm_concepts', 'llm_level', 'llm_applications', 'llm_summary'])):
                    
                    content.append(r"\begin{aibox}[title=AI Analysis]")
                    
                    if talk.get('llm_concepts'):
                        content.append(f"\\textbf{{Key Concepts:}} {self.escape_latex(talk.get('llm_concepts'))}\\\\")
                    if talk.get('llm_level'):
                        content.append(f"\\textbf{{Difficulty Level:}} {self.escape_latex(talk.get('llm_level'))}\\\\")
                    if talk.get('llm_applications'):
                        content.append(f"\\textbf{{Applications:}} {self.escape_latex(talk.get('llm_applications'))}\\\\")
                    if talk.get('llm_summary'):
                        content.append(f"\\textbf{{Summary:}} {self.escape_latex(talk.get('llm_summary'))}")
                    
                    content.append(r"\end{aibox}")
                    content.append("\n\n")
                
                # Now include the original LaTeX content
                if talk_id:
                    tex_file_path = f"{abstracts_base_path}/{talk_id}.tex"
                    
                    import os
                    if os.path.exists(tex_file_path):
                        # Extract just the \begin{talk}...\end{talk} content
                        talk_content = self.extract_talk_content_from_file(tex_file_path)
                        if talk_content:
                            content.append(f"% Including abstract content from: {talk_id}.tex\n")
                            content.append(talk_content)
                            content.append("\n\\newpage\n\n")
                        else:
                            content.append(f"% Could not extract talk content from: {talk_id}.tex\n")
                            self.add_fallback_talk_info(content, talk, talk_id)
                    else:
                        content.append(f"% File not found: {talk_id}.tex\n")
                        self.add_fallback_talk_info(content, talk, talk_id)
                else:
                    content.append(f"% No ID available for this talk\n")
                    self.add_fallback_talk_info(content, talk, None)
                
            except Exception as e:
                content.append(f"% ERROR processing AI talk {i+1}: {str(e)}\n")
                content.append(f"\\section{{Talk {i+1} - Processing Error}}\n\n")
                content.append("\\textit{Error processing this talk's information.}\n\n")
                content.append("\\newpage\n\n")
        
        return ''.join(content)
    
    def create_latex_document(self, selected_talks: List[Dict], include_abstracts: bool, 
                            include_schedule: bool, include_conflicts: bool) -> str:
        """Create the LaTeX document content"""
        doc_parts = []
        
        # Document header
        doc_parts.append(r"""
\documentclass[12pt,letterpaper,oneside]{book}

\usepackage{needspace}
\usepackage[dvipsnames]{xcolor}
\usepackage{xurl}
\usepackage[colorlinks=true,linkcolor=NavyBlue,citecolor=ForestGreen,urlcolor=Magenta]{hyperref}
\hypersetup{breaklinks=true}
\usepackage{bm}
\usepackage{verbatim}
\usepackage{tabularx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathrsfs}
\usepackage{enumitem}
\setlist[itemize]{itemsep=2pt, parsep=0pt}

\usepackage{fancyhdr}
\usepackage{datetime}

\newdateformat{myformat}{\twodigit{\THEDAY}~\monthname[\THEMONTH]~\THEYEAR}

\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.4pt}

\fancyfoot[C]{%
    \makebox[\textwidth]{%
        \myformat\today\ \currenttime\ \textcolor{gray}{\footnotesize (""" + f"{self.llm.model_name if self.llm.available else 'N/A'}" + r""")}
        \hfill
        \thepage
    }
}

\makeatletter
\renewcommand{\footrule}{%
    \hrule width \textwidth height \footrulewidth \vskip2pt
}
\makeatother

\fancypagestyle{plain}{%
    \fancyhf{}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0.4pt}
\fancyfoot[C]{%
    \makebox[\textwidth]{%
        \myformat\today\ \currenttime\ \textcolor{gray}{\footnotesize (""" + f"{self.llm.model_name if self.llm.available else 'N/A'}" + r""")}
        \hfill
        \thepage
    }
}
}

\pagestyle{fancy}

\usepackage[top=1in,bottom=1in,left=1in,right=1in,headheight=0pt,headsep=0pt,footskip=30pt]{geometry}

\begin{document}
""")
        
        # Title page
        doc_parts.append(r"""
\begin{titlepage}
\centering
\vspace*{2cm}

{\Huge\bfseries MCM 2025}\\[0.5cm]
{\LARGE Personalized Program Book}\\[1.5cm]

{\Large The Fifteenth International Conference on\\
Monte Carlo and Applications}\\[1cm]

{\large Illinois Institute of Technology\\
Chicago, Illinois, USA\\[1cm]
July 28 -- August 1, 2025}\\[2cm]

{\large Generated on \myformat\today}\\[0.5cm]
{\footnotesize AI Model: \textcolor{gray}{\texttt{""" + f"{self.llm.model_name if self.llm.available else 'N/A'}" + r"""}}}\\[1cm]

""")
        
        # Add user interests summary
        if st.session_state.user_interests:
            interests = st.session_state.user_interests
            doc_parts.append(r"\begin{flushleft}")
            doc_parts.append(r"\textbf{Your Research Interests:}\\[0.5cm]")
            
            if interests.get('keywords'):
                keywords_str = ', '.join(interests['keywords'])
                doc_parts.append(f"\\textbf{{Keywords:}} {self.escape_latex(keywords_str)}\\\\[0.3cm]")
            
            if interests.get('areas'):
                areas_str = ', '.join(interests['areas'])
                doc_parts.append(f"\\textbf{{Research Areas:}} {self.escape_latex(areas_str)}\\\\[0.3cm]")

            if interests.get('user_preferences_only'):
                preferences_str = interests['user_preferences_only']
                doc_parts.append(f"\\textbf{{Preferences:}} {self.escape_latex(preferences_str)}\\\\[0.3cm]")
            
            doc_parts.append(f"\\textbf{{Selected Talks:}} {len(selected_talks)}\\\\")
            doc_parts.append(r"\end{flushleft}")
        
        doc_parts.append(r"\end{titlepage}")
        
        # Table of contents
        doc_parts.append(r"""
\tableofcontents
\newpage
""")
        
        # Schedule section
        if include_schedule:
            doc_parts.append(self.create_schedule_section(selected_talks))
        
        # Conflicts warning
        if include_conflicts:
            conflicts = self.check_conflicts(selected_talks)
            if conflicts:
                doc_parts.append(self.create_conflicts_section(conflicts))
        
        # Abstracts section - Use only the selected talks, not all abstract talks
        if include_abstracts:
            doc_parts.append(self.create_abstracts_section(selected_talks))
        
        doc_parts.append(r"\end{document}")
        
        return ''.join(doc_parts)
    
    def create_schedule_section(self, selected_talks: List[Dict]) -> str:
        """Create personalized schedule section using MCM schedule format with AI-selected sessions"""
        content = [r"""
\chapter{Your Personalized Schedule}

This section contains your personalized conference schedule with AI-selected parallel sessions that best match your interests.

"""]
        
        # Create personalized schedule for each day/session period
        schedule_periods = [
            ('Mon', 'Morning', 'MonMorning'),
            ('Mon', 'Afternoon', 'MonAfternoon'), 
            ('Tue', 'Morning', 'TueMorning'),
            ('Tue', 'Afternoon', 'TueAfternoon'),
            ('Wed', 'Morning', 'WedMorning'),
            ('Wed', 'Afternoon', 'WedAfternoon'),
            ('Thu', 'Morning', 'ThuMorning'),
            ('Thu', 'Afternoon', 'ThuAfternoon'),
            ('Fri', 'Morning', 'FriMorning')
        ]
        
        for day, period, label in schedule_periods:
            period_content = self.create_schedule_period(day, period, label, selected_talks)
            if period_content:
                content.append(period_content)
        
        return ''.join(content)
     
    
    def discover_parallel_sessions_for_period(self, day: str, period: str) -> List[str]:
        """Hardcoded parallel session mappings from SessionList.csv """
        
        # Hardcoded session mappings based on SessionList.csv data 
        session_mappings = {
            ('Monday', 'Morning'): ['S1', 'S2', 'S3', 'S4', 'T1'],
            ('Monday', 'Afternoon'): ['S5', 'S6', 'S7', 'T4', 'T12'],
            ('Tuesday', 'Morning'): ['S8', 'S9', 'S10', 'S11', 'T2'],
            ('Tuesday', 'Afternoon'): ['S12', 'S13', 'S14', 'S15', 'T5'],
            ('Wednesday', 'Morning'): ['S16', 'S17', 'S18', 'T15', 'T6'],
            ('Wednesday', 'Afternoon'): ['S19', 'S20', 'S21', 'T16'],  # Only 4 sessions in afternoon
            ('Thursday', 'Morning'): ['S22', 'S23', 'S24', 'T8', 'T13'],
            ('Thursday', 'Afternoon'): ['S25', 'S26', 'S27', 'T7', 'T11'],
            ('Friday', 'Morning'): ['S28', 'S29', 'T3', 'T9']  # Only 4 sessions on Friday
        }
        
        # Get the sessions for this day/period
        key = (day, period)
        sessions = session_mappings.get(key, [])
        
        if sessions:
            print(f"DEBUG: Hardcoded sessions for {day} {period}: {sessions}")
        else:
            print(f"WARNING: No hardcoded sessions found for {day} {period}")
        
        return sessions
    
    def get_dynamic_session_map(self) -> Dict:
        """Dynamically load session information from LaTeX files"""
        session_map = {}
        
        # Define paths for different file types
        out_path = self.base_path / "out"
        abstracts_path = self.base_path / "input" / "abstracts"
        
        # Find all session files (sessS*.tex, sessT*.tex) in out directory
        if out_path.exists():
            for sess_file in out_path.glob("sess[ST]*.tex"):
                try:
                    # Extract session ID from filename (e.g., "sessS4.tex" -> "S4")
                    session_id = sess_file.stem.replace("sess", "")
                    
                    # Load schedule/timing information from sessS4.tex
                    schedule_info = self.parse_session_schedule_file(sess_file)
                    
                    # Load session description and organizers from S4.tex
                    session_desc_file = abstracts_path / f"{session_id}.tex"
                    session_info = {}
                    if session_desc_file.exists():
                        session_info = self.parse_session_description_file(session_desc_file)
                    
                    # Load individual talk abstracts (S4-1.tex, S4-2.tex, etc.)
                    talks = self.load_session_talk_abstracts(session_id, abstracts_path)
                    
                    # Combine all information
                    session_map[session_id] = {
                        'title': session_info.get('title', f'Session {session_id}'),
                        'organizer': session_info.get('primary_organizer', 'Unknown'),
                        'organizers': session_info.get('organizers', []),
                        'description': session_info.get('description', ''),
                        'time': schedule_info.get('time', 'TBD'),
                        'room': schedule_info.get('room', 'TBD'),
                        'talks': talks,
                        'talk_count': len(talks)
                    }
                    
                    #print(f"Loaded session {session_id}: {session_map[session_id]['title']} - {session_map[session_id]['organizer']}")
                    
                except Exception as e:
                    print(f"Error loading session {sess_file.name}: {e}")
        
        return session_map
    
    def parse_session_schedule_file(self, sess_file: Path) -> Dict:
        """Parse sessS4.tex file for timing and room information"""
        try:
            content = self.safe_read_file(sess_file)
            if not content:
                return {}
            
            info = {}
            
            # Extract timing information: \timeslot{Mon, Jul 28, 2025--Morning}{10:30}{12:30}
            time_match = re.search(r'\\timeslot\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}', content)
            if time_match:
                date_desc = time_match.group(1)
                start_time = time_match.group(2)
                end_time = time_match.group(3)
                info['time'] = f"{date_desc}, {start_time}-{end_time}"
            
            # Extract room information from the same line or nearby
            room_match = re.search(r'\{([^}]*(?:Auditorium|Room|Hall|WH|HH)[^}]*)\}', content)
            if room_match:
                info['room'] = room_match.group(1).strip()
            
            # Extract talk list
            talk_matches = re.findall(r'\\sessionTalk\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}', content)
            info['talks_schedule'] = []
            for title, speaker, talk_id in talk_matches:
                info['talks_schedule'].append({
                    'title': title.strip(),
                    'speaker': speaker.strip(),
                    'talk_id': talk_id.strip()
                })
            
            return info
            
        except Exception as e:
            print(f"Error parsing session schedule file {sess_file}: {e}")
            return {}
    
    def parse_session_description_file(self, session_file: Path) -> Dict:
        """Parse S4.tex file for session description and organizers"""
        try:
            content = self.safe_read_file(session_file)
            if not content:
                return {}
            
            info = {}
            
            # Extract session title and organizers using the session environment
            session_match = re.search(r'\\begin\{session\}(.*?)\\end\{session\}', content, re.DOTALL)
            if session_match:
                session_content = session_match.group(1).strip()
                
                # Extract parameters from the session environment
                params = self.extract_parameters_from_content(session_content, 5)
                
                if len(params) >= 1:
                    info['title'] = params[0].strip()
                
                # Extract organizers using \organizer{name}{affiliation}{email}
                organizer_matches = re.findall(r'\\organizer\{([^}]*)\}\s*\{([^}]*)\}\s*\{([^}]*)\}', content)
                organizers = []
                for name, affiliation, email in organizer_matches:
                    organizers.append({
                        'name': name.strip(),
                        'affiliation': affiliation.strip(),
                        'email': email.strip()
                    })
                
                info['organizers'] = organizers
                info['primary_organizer'] = organizers[0]['name'] if organizers else 'Unknown'
                
                # Extract session description (text after parameters but before \end{session})
                desc_start = session_match.end()
                desc_end = content.find('\\end{session}', desc_start)
                if desc_end != -1:
                    desc_content = content[desc_start:desc_end].strip()
                    # Clean up the description
                    desc_lines = []
                    for line in desc_content.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('%') and not line.startswith('\\'):
                            desc_lines.append(line)
                    info['description'] = '\n'.join(desc_lines)
            
            return info
            
        except Exception as e:
            print(f"Error parsing session description file {session_file}: {e}")
            return {}
    
    def load_session_talk_abstracts(self, session_id: str, abstracts_path: Path) -> List[Dict]:
        """Load individual talk abstracts for a session (S4-1.tex, S4-2.tex, etc.)"""
        talks = []
        
        try:
            # Find all talk files for this session
            talk_files = list(abstracts_path.glob(f"{session_id}-*.tex"))
            
            for talk_file in sorted(talk_files):
                try:
                    content = self.safe_read_file(talk_file)
                    if content:
                        talk_data = self.parse_talk_abstract(content, talk_file.stem)
                        if talk_data:
                            talks.append(talk_data)
                
                except Exception as e:
                    print(f"Error loading talk abstract {talk_file.name}: {e}")
        
        except Exception as e:
            print(f"Error loading talk abstracts for session {session_id}: {e}")
        
        return talks
    
    def ai_select_best_session(self, session_ids: List[str], day: str, period: str, selected_talks: List[Dict]) -> str:
        """Use AI to select the best session from parallel options based on user interests"""
        
        # DEBUG: Print all inputs to ai_select_best_session
        print("\n" + "="*80)
        print("DEBUG: ai_select_best_session() INPUTS:")
        print("="*80)
        print(f"session_ids: {session_ids} (count: {len(session_ids) if session_ids else 0})")
        print(f"day: '{day}'")
        print(f"period: '{period}'")
        print(f"selected_talks count: {len(selected_talks) if selected_talks else 0}")
        
        if selected_talks:
            print("selected_talks details:")
            for i, talk in enumerate(selected_talks[:3]):  # Show first 3 talks
                print(f"  {i+1}. Title: '{talk.get('title', 'Unknown')}'")
                print(f"     Speaker: '{talk.get('speaker', 'Unknown')}'")
                print(f"     Type: '{talk.get('type', 'Unknown')}'")
        
        user_interests = st.session_state.get('user_interests', {})
        print(f"\nuser_interests from session state:")
        print(f"  keywords: {user_interests.get('keywords', [])}")
        print(f"  areas: {user_interests.get('areas', [])}")
        print(f"  experience: '{user_interests.get('experience', 'Not specified')}'")
        print(f"  preferences: '{user_interests.get('preferences', 'None')[:200]}{'...' if len(user_interests.get('preferences', '')) > 200 else ''}'")
        print("="*80)
        
        if not self.llm.available:
            # Fallback: select first session
            #print(f"DEBUG: LLM not available, selecting first session {session_ids[0] if session_ids else 'None'}")
            return session_ids[0] if session_ids else None
            
        # Get user interests
        user_interests = st.session_state.get('user_interests', {})
        if not user_interests:
            #print(f"DEBUG: No user interests found, selecting first session {session_ids[0] if session_ids else 'None'}")
            return session_ids[0] if session_ids else None
            
        #print(f"DEBUG: AI selecting best session for {day} {period} from options: {session_ids}")
        #print(f"DEBUG: User interests keywords: {user_interests.get('keywords', [])}")
        #print(f"DEBUG: User preferences: {user_interests.get('preferences', '')[:200]}...")
            
        # Load session talk files for AI comparison
        session_options = []
        
        print(f"DEBUG: Loading session talk files for AI comparison:")
        for i, session_id in enumerate(session_ids):
            print(f"  Loading session {i+1}/{len(session_ids)}: {session_id}")
            
            # Load the consolidated session talk file
            session_talk_file = self.base_path / "out" / f"{session_id}_sess_talks.tex"
            
            if session_talk_file.exists():
                try:
                    # Read the full LaTeX content of the session talks
                    session_talks_content = self.safe_read_file(session_talk_file)
                    
                    if session_talks_content:
                        # Extract session title from the first line (e.g., \section{Track A: Stochastic Computation and Complexity, Part I})
                        title_match = re.search(r'\\section\{([^}]+)\}', session_talks_content)
                        title = title_match.group(1) if title_match else f"Session {session_id}"
                        
                        # Count the number of talks in the file
                        talk_count = len(re.findall(r'\\begin\{talk\}', session_talks_content))
                        
                        # Extract a brief sample of talk titles for debugging
                        talk_titles = re.findall(r'\\begin\{talk\}\s*\{([^}]+)\}', session_talks_content)
                        
                        session_option = {
                            'id': session_id,
                            'title': title,
                            'latex_content': session_talks_content,
                            'talk_count': talk_count,
                            'talk_titles': talk_titles[:3]  # First 3 talk titles for debugging
                        }
                        
                        # DEBUG: Show what data will be sent to AI for this session
                        print(f"    ✓ {session_id}: '{title}'")
                        print(f"      LaTeX file: {session_talk_file.name} ({len(session_talks_content)} chars)")
                        print(f"      Number of talks: {talk_count}")
                        if talk_titles:
                            print(f"      Sample talk titles: {talk_titles[:2]}")  # Show first 2 talk titles
                        
                        session_options.append(session_option)
                    else:
                        print(f"    ✗ {session_id}: Empty session talk file")
                        
                except Exception as e:
                    print(f"    ✗ {session_id}: Error reading session talk file: {e}")
            else:
                print(f"    ✗ {session_id}: Session talk file not found at {session_talk_file}")
                
        if not session_options:
            #print(f"DEBUG: No session options loaded, selecting first session {session_ids[0] if session_ids else 'None'}")
            return session_ids[0] if session_ids else None
            
        # Use AI to evaluate sessions
        try:
            time_slot = f"{day} {period}"
            #print(f"DEBUG: Calling AI to select from {len(session_options)} sessions for {time_slot}")
            
            # Show summary of what we're sending to AI
            print(f"DEBUG: Session summaries being sent to AI:")
            for i, session in enumerate(session_options):
                title = session.get('title', 'No title')
                
                if session.get('latex_content'):
                    content_size = len(session['latex_content'])
                    talk_count = session.get('talk_count', 0)
                    latex_indicator = "📄 Full LaTeX"
                    details = f"{talk_count} talks, {content_size} chars"
                
                print(f"  {i+1}. {session['id']}: '{title}' ({latex_indicator} - {details})")
            
            #print(f"DEBUG: User keywords for matching: {user_interests.get('keywords', [])}")
            
            # Pre-analyze keyword matches to help AI make better decisions
            keyword_matches = {}
            user_keywords = [kw.lower().strip() for kw in user_interests.get('keywords', [])]
            
            for session in session_options:
                session_id = session['id']
                title = session.get('title', '').lower()
                
                # For sessions with LaTeX content, search within the full content
                if session.get('latex_content'):
                    # Remove LaTeX commands for better text matching
                    clean_content = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*', ' ', session['latex_content'])
                    clean_content = re.sub(r'[{}\\]', ' ', clean_content)
                    search_text = f"{title} {clean_content.lower()}"
                else:
                    # Fallback for sessions without LaTeX content
                    description = session.get('description', '').lower()
                    organizers_text = ' '.join([str(org).lower() for org in session.get('organizers', [])])
                    talks_text = ' '.join(session.get('talks', [])).lower()
                    search_text = f"{title} {description} {organizers_text} {talks_text}"
                
                matches = []
                for keyword in user_keywords:
                    if keyword in search_text:
                        matches.append(keyword)
                
                keyword_matches[session_id] = {
                    'matches': matches,
                    'score': len(matches),
                    'title': session.get('title', 'Unknown'),
                    'has_latex': session.get('latex_content') is not None
                }
                
                latex_indicator = "📄 LaTeX" if session.get('latex_content') else "📝 Basic"
                print(f"  Keywords found in {session_id} ({latex_indicator}) ('{title[:50]}...'): {matches} (score: {len(matches)})")
            
            # Show which session has the best keyword match
            best_match = max(keyword_matches.items(), key=lambda x: x[1]['score'])
            #print(f"DEBUG: Best keyword match: {best_match[0]} with score {best_match[1]['score']} - '{best_match[1]['title']}'")
            
            # Add explicit guidance to help AI choose correctly
            enhanced_user_interests = user_interests.copy()
            enhanced_user_interests['keyword_analysis'] = keyword_matches

            enhanced_user_interests['ai_guidance'] = f"IMPORTANT: User is specifically interested in keywords: {user_keywords}. " \
                                                       f"Session {best_match[0]} ('{best_match[1]['title']}') has the highest keyword match score of {best_match[1]['score']}. " \
                                                       f"Please prioritize sessions that contain the user's keywords in their titles and descriptions."
            
            best_session_data = self.llm.select_best_parallel_session(session_options, enhanced_user_interests, time_slot)
            selected_id = best_session_data.get('id', session_ids[0]) if best_session_data else session_ids[0]
            reasoning = best_session_data.get('ai_selection_reasoning', 'No reasoning provided') if best_session_data else 'Selection failed'
            #print(f"DEBUG: *** AI SELECTED SESSION {selected_id} ***")
            #print(f"DEBUG: AI reasoning: {reasoning}")
            
            # Show which session was selected and why it matched the user's interests
            for session in session_options:
                if session['id'] == selected_id:
                    #print(f"DEBUG: Selected session details:")
                    print(f"  Title: {session.get('title', 'Unknown')}")
                    print(f"  Organizers: {session.get('organizers', [])}")
                    print(f"  Keywords match check:")
                    title_lower = session.get('title', '').lower()
                    for keyword in user_interests.get('keywords', []):
                        if keyword.lower() in title_lower:
                            print(f"    ✓ Keyword '{keyword}' found in title")
                    break
            
            return selected_id
        except Exception as e:
            print(f"ERROR: AI session selection failed: {e}")
            return session_ids[0] if session_ids else None
    
    def load_session_data(self, session_id: str) -> Dict:
        """Load session data dynamically from session files and abstract files"""
        # Get dynamic session data from actual LaTeX files
        session_map = self.get_dynamic_session_map()
        
        session_data = {
            'id': session_id,
            'title': '',
            'organizer': '',
            'description': '',
            'talks': []
        }
        
        # Use hardcoded data if available
        if session_id in session_map:
            session_data.update(session_map[session_id])
        
        # Try to load from session abstract file (e.g., S1.tex, S2.tex)
        mcm_path = self.base_path.parent / "MCM_ProgramBook_TEX"
        session_file = mcm_path / f"sess{session_id}.tex"
        
        if session_file.exists():
            session_info = self.parse_session_file(session_file)
            session_data.update(session_info)
        
        # Load session abstract from preprocess/input/abstracts
        abstract_file = self.abstracts_path / f"{session_id}.tex"
        if abstract_file.exists():
            try:
                abstract_content = self.safe_read_file(abstract_file)
                if abstract_content:
                    session_abstract = self.parse_session_abstract(abstract_content, session_id)
                    session_data.update(session_abstract)
                    ##print(f"DEBUG: Loaded session abstract for {session_id}: title='{session_data.get('title', 'N/A')}'")
            except Exception as e:
                print(f"Error loading session abstract {session_id}: {e}")
        
        # CRITICAL: Also try to load from main consolidated files where S4 might be defined
        if not session_data.get('title') or session_data.get('title') == '':
            consolidated_file = self.base_path.parent / "MCM_ProgramBook_TEX" / "MCM2025_consolidated.tex"
            if consolidated_file.exists():
                try:
                    consolidated_content = self.safe_read_file(consolidated_file)
                    if consolidated_content:
                        # Look for this session in the consolidated file
                        session_pattern = rf'\\begin\{{session\}}.*?{re.escape(session_id)}.*?\\end\{{session\}}'
                        session_match = re.search(session_pattern, consolidated_content, re.DOTALL | re.IGNORECASE)
                        if session_match:
                            session_content = session_match.group(0)
                            parsed_session = self.parse_session_from_main_file(session_content, session_id)
                            if parsed_session:
                                session_data.update(parsed_session)
                                #print(f"DEBUG: Loaded session from consolidated file for {session_id}: title='{session_data.get('title', 'N/A')}'")
                except Exception as e:
                    print(f"Error loading session from consolidated file for {session_id}: {e}")
            
            # Also try special session submissions file
            special_sessions_file = self.base_path.parent / "MCM_ProgramBook_TEX" / "special_session_submissions_talks.tex"
            if special_sessions_file.exists() and (not session_data.get('title') or session_data.get('title') == ''):
                try:
                    special_content = self.safe_read_file(special_sessions_file)
                    if special_content:
                        # Look for this session in the special sessions file
                        session_pattern = rf'\\begin\{{session\}}.*?{re.escape(session_id)}.*?\\end\{{session\}}'
                        session_match = re.search(session_pattern, special_content, re.DOTALL | re.IGNORECASE)
                        if session_match:
                            session_content = session_match.group(0)
                            parsed_session = self.parse_session_from_main_file(session_content, session_id)
                            if parsed_session:
                                session_data.update(parsed_session)
                                #print(f"DEBUG: Loaded session from special sessions file for {session_id}: title='{session_data.get('title', 'N/A')}'")
                except Exception as e:
                    print(f"Error loading session from special sessions file for {session_id}: {e}")
        
        # Load individual talk abstracts (e.g., S1-1.tex, S1-2.tex, etc.)
        for i in range(1, 5):  # Typically 4 talks per session
            talk_id = f"{session_id}-{i}"
            talk_file = self.abstracts_path / f"{talk_id}.tex"
            if talk_file.exists():
                try:
                    talk_content = self.safe_read_file(talk_file)
                    if talk_content:
                        talk_data = self.parse_talk_abstract(talk_content, talk_id)
                        if talk_data:
                            session_data['talks'].append(talk_data)
                except Exception as e:
                    print(f"Error loading talk abstract {talk_id}: {e}")
        
        # Final debug output for this session
        #print(f"DEBUG: Final session data for {session_id}:")
        print(f"  Title: '{session_data.get('title', 'N/A')}'")
        print(f"  Organizers: {session_data.get('organizers', 'N/A')}")
        print(f"  Description length: {len(session_data.get('description', ''))}")
        print(f"  Number of talks: {len(session_data.get('talks', []))}")
        
        return session_data
    
    def parse_session_file(self, session_file: Path) -> Dict:
        """Parse session file (e.g., sessS4.tex) to extract session info"""
        try:
            content = self.safe_read_file(session_file)
            if not content:
                return {}
            
            session_info = {}
            
            # Extract time slot info
            timeslot_match = re.search(r'\\timeslot\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}', content)
            if timeslot_match:
                session_info['time_period'] = timeslot_match.group(1)
                session_info['start_time'] = timeslot_match.group(2)
                session_info['end_time'] = timeslot_match.group(3)
                session_info['room'] = timeslot_match.group(4)
            
            # Extract session talks
            talk_matches = re.findall(r'\\sessionTalk\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}', content)
            talks = []
            for title, speaker, talk_id in talk_matches:
                talks.append({
                    'id': talk_id.strip(),
                    'title': title.strip(),
                    'speaker': speaker.strip()
                })
            session_info['talks'] = talks
            
            return session_info
            
        except Exception as e:
            print(f"Error parsing session file {session_file}: {e}")
            return {}
    
    def get_session_info(self, session_id: str, day: str, period: str) -> str:
        """Get formatted session info for the selected session"""
        
        # Get dynamic session data from actual LaTeX files
        session_map = self.get_dynamic_session_map()
        
        if session_id not in session_map:
            return ""
            
        session_info = session_map[session_id]
        
        content = []
        content.append("\\rowcolor{\\SessionTitleColor}\n")
        content.append(f"&\\tableSpecialCL{{{session_info['room']}}}\n")
        content.append(f"{{{session_info['title']}}}\n")
        content.append(f"{{{session_id}}}\n") 
        content.append(f"{{{session_info['organizer']}}}\n")
        content.append("\\\\\\hline\n\n")
        
        return ''.join(content)
    
    def create_conflicts_section(self, conflicts: Dict) -> str:
        """Create conflicts warning section"""
        if not conflicts:
            return ""
        
        content = [r"""
\chapter{Schedule Conflicts Warning}

\textcolor{red}{\textbf{Warning: The following schedule conflicts were detected in your selection:}}

"""]
        
        for time, talks in conflicts.items():
            content.append(f"\\subsection{{Conflict at {self.escape_latex(time)}}}\n\n")
            content.append("The following talks are scheduled at the same time:\n\n")
            content.append(r"\begin{itemize}" + "\n")
            
            for talk in talks:
                title = self.escape_latex(talk.get('title', ''))
                speaker = self.escape_latex(talk.get('speaker', ''))
                content.append(f"\\item \\textbf{{{title}}} -- {speaker}\n")
            
            content.append(r"\end{itemize}" + "\n\n")
        
        return ''.join(content)
    
    def create_abstracts_section(self, selected_talks: List[Dict]) -> str:
        """Create the abstracts section using LaTeX content from main files and fallback to individual files"""
        content = [r"""
\chapter{Selected Talk Abstracts}
"""]
        
        # Sort talks by relevance score
        sorted_talks = sorted(selected_talks, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        for i, talk in enumerate(sorted_talks):
            try:
                # Get the talk ID
                talk_id = talk.get('id', '')
                talk_type = talk.get('type', 'unknown')
                
                # Add debug info
                content.append(f"% DEBUG: Talk {i+1}: ID = '{talk_id}', Type = '{talk_type}'\n")
                content.append(f"% DEBUG: Title = '{talk.get('title', 'NO_TITLE')}'\n")
                content.append(f"% DEBUG: Speaker = '{talk.get('speaker', 'NO_SPEAKER')}'\n")
                content.append(f"% DEBUG: Session ID = '{talk.get('session_id', 'NO_SESSION_ID')}'\n")
                content.append(f"% DEBUG: Abstract = '{talk.get('abstract', 'NO_ABSTRACT')}'\n")
                content.append(f"% DEBUG: Affliation = '{talk.get('affiliation', 'NO_AFFILIATION')}'\n")
                #content.append(f"% DEBUG: Source = '{talk.get('source', 'NO_SOURCE')}'\n")
                #content.append(f"% DEBUG: Relevance Score: {talk.get('relevance_score', 0):.1f}%\n")
                
                if talk_type == 'session':
                    # Handle session display - now using direct \input for session and its talks
                    session_id = talk.get('id', '')
                    title = self.escape_latex(talk.get('title', f'Session {i+1}'))
                    
                    # Create section for the session
                    content.append(f"\\section{{Session {i+1} – {session_id}: {title}}}\n\n")
                    
                    # Input the session schedule file from MCM_ProgramBook_TEX
                    content.append(f"% Including session schedule file via \\input\n")
                    content.append(f"\\input{{{session_id}_sess_talks}}\n\n")
   
                    content.append("\n\\newpage\n\n")
                
                elif talk_id and st.session_state.abstracts_data.get(talk_id):
                    # We have comprehensive data from main LaTeX files
                    abstract_data = st.session_state.abstracts_data[talk_id]
                    
                    # Always use properly formatted comprehensive talk info instead of raw LaTeX
                    # to avoid concatenated text issues like "TitleSpeakerAffiliationEmail"
                    content.append(f"% Using comprehensive data from main LaTeX files (formatted)\n")
                    
                    # If it's a session, extract session content
                    if abstract_data.get('type') == 'session':
                        session_id = abstract_data.get('id', '')
                        title = self.escape_latex(abstract_data.get('title', f'Session {i+1}'))
                        
                        # Create section for the session
                        content.append(f"\\section{{Session {i+1} – {session_id}: {title}}}\n\n")
                        
                        # Input the session schedule file
                        content.append(f"\\input{{{session_id}_sess_talks}}\n\n")
                        
                        
                        content.append("\n\\newpage\n\n")
                    else:
                        # For talks, always use the formatted version to avoid concatenation issues
                        self.add_comprehensive_talk_info(content, abstract_data, i+1)
                
                elif talk_id:
                    # Fallback to individual files in preprocess/input/abstracts/
                    abstracts_base_path = "/Users/terrya/Documents/ProgramData/MCM-2025-Program/preprocess/input/abstracts"
                    tex_file_path = f"{abstracts_base_path}/{talk_id}.tex"
                    
                    content.append(f"% DEBUG: Fallback to individual file: {tex_file_path}\n")
                    
                    import os
                    if os.path.exists(tex_file_path):
                        content.append(f"% DEBUG: Individual file exists: YES\n")
                        talk_content = self.extract_talk_content_from_file(tex_file_path)
                        if talk_content:
                            content.append(f"% Including abstract content from individual file: {talk_id}.tex\n")
                            content.append(talk_content)
                            content.append("\n\\newpage\n\n")
                        else:
                            content.append(f"% Could not extract talk content from individual file: {talk_id}.tex\n")
                            self.add_fallback_talk_info(content, talk, talk_id)
                    else:
                        content.append(f"% Individual file not found: {talk_id}.tex\n")
                        self.add_fallback_talk_info(content, talk, talk_id)
                else:
                    # No ID available
                    content.append(f"% No ID available for this talk\n")
                    self.add_fallback_talk_info(content, talk, None)
                
            except Exception as e:
                # If there's an error with this talk, add a safe placeholder
                content.append(f"% ERROR processing talk {i+1}: {str(e)}\n")
                content.append(f"\\section{{Talk {i+1} - Processing Error}}\n\n")
                content.append("\\textit{Error processing this talk's information.}\n\n")
                content.append("\\newpage\n\n")
        
        return ''.join(content)
    
    def extract_talk_content_from_file(self, file_path: str) -> str:
        """Extract the \\begin{talk}...\\end{talk} or \\begin{session}...\\end{session} content from a LaTeX file"""
        try:
            content = self.safe_read_file(file_path)
            if not content:
                return None
            
            if IS_DEBUG_APP:
                print(f"DEBUG: Reading file {file_path}, content length: {len(content)}")
            
            # First try to find \begin{talk} and \end{talk} boundaries
            start_match = re.search(r'\\begin\{talk\}', content)
            end_match = re.search(r'\\end\{talk\}', content)
            
            if start_match and end_match:
                # Extract the talk content including the begin/end tags
                talk_content = content[start_match.start():end_match.end()]
                if IS_DEBUG_APP:
                    print(f"DEBUG: Found talk boundaries, extracted content length: {len(talk_content)}")
                return talk_content
            
            # If no talk boundaries, try session boundaries
            start_match = re.search(r'\\begin\{session\}', content)
            end_match = re.search(r'\\end\{session\}', content)
            
            if start_match and end_match:
                # Extract the session content including the begin/end tags
                session_content = content[start_match.start():end_match.end()]
                if IS_DEBUG_APP:
                    print(f"DEBUG: Found session boundaries, extracted content length: {len(session_content)}")
                return session_content
            
            # If neither found, log the issue
            if IS_DEBUG_APP:
                print(f"DEBUG: Could not find talk or session boundaries in {file_path}")
            return None
                
        except Exception as e:
            print(f"ERROR: Error reading file {file_path}: {e}")
            return None

    def extract_abstract_text_from_latex(self, latex_content: str) -> str:
        """Extract just the abstract text from LaTeX talk or session content"""
        if not latex_content:
            return ""
        
        try:
            # First, try to extract text between \begin{talk} and \end{talk}
            talk_match = re.search(r'\\begin\{talk\}(.*?)\\end\{talk\}', latex_content, re.DOTALL)
            if talk_match:
                talk_content = talk_match.group(1)
                
                # Extract the 6 parameters first to find where abstract text starts
                params = []
                i = 0
                while i < len(talk_content) and len(params) < 6:
                    # Skip whitespace and comments
                    while i < len(talk_content) and (talk_content[i].isspace() or talk_content[i] == '%'):
                        if talk_content[i] == '%':
                            # Skip to end of line
                            while i < len(talk_content) and talk_content[i] != '\n':
                                i += 1
                        i += 1
                    
                    if i >= len(talk_content):
                        break
                        
                    # Look for opening brace
                    if talk_content[i] == '{':
                        brace_count = 1
                        start = i + 1
                        i += 1
                        
                        # Find matching closing brace
                        while i < len(talk_content) and brace_count > 0:
                            if talk_content[i] == '{':
                                brace_count += 1
                            elif talk_content[i] == '}':
                                brace_count -= 1
                            i += 1
                        
                        if brace_count == 0:
                            param = talk_content[start:i-1].strip()
                            params.append(param)
                        else:
                            break
                    else:
                        i += 1
                
                # Now extract text after the 6th parameter
                if len(params) >= 6 and i < len(talk_content):
                    remaining_text = talk_content[i:].strip()
                    
                    # Clean up the abstract text
                    lines = remaining_text.split('\n')
                    clean_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('%'):
                            clean_lines.append(line)
                    
                    abstract_text = ' '.join(clean_lines)
                    # Clean up extra whitespace
                    abstract_text = re.sub(r'\s+', ' ', abstract_text).strip()
                    
                    return abstract_text
            
            # If no talk found, try session format
            session_match = re.search(r'\\begin\{session\}(.*?)\\end\{session\}', latex_content, re.DOTALL)
            if session_match:
                session_content = session_match.group(1)
                
                # Extract session parameters (usually fewer than talk parameters)
                params = []
                i = 0
                while i < len(session_content) and len(params) < 9:  # Sessions can have up to 9 parameters
                    # Skip whitespace and comments
                    while i < len(session_content) and (session_content[i].isspace() or session_content[i] == '%'):
                        if session_content[i] == '%':
                            while i < len(session_content) and session_content[i] != '\n':
                                i += 1
                        i += 1
                    
                    if i >= len(session_content):
                        break
                        
                    # Look for opening brace
                    if session_content[i] == '{':
                        brace_count = 1
                        start = i + 1
                        i += 1
                        
                        # Find matching closing brace
                        while i < len(session_content) and brace_count > 0:
                            if session_content[i] == '{':
                                brace_count += 1
                            elif session_content[i] == '}':
                                brace_count -= 1
                            i += 1
                        
                        if brace_count == 0:
                            param = session_content[start:i-1].strip()
                            params.append(param)
                        else:
                            break
                    else:
                        i += 1
                
                # Extract description text after parameters
                if i < len(session_content):
                    remaining_text = session_content[i:].strip()
                    
                    # Clean up the description text
                    lines = remaining_text.split('\n')
                    clean_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('%'):
                            clean_lines.append(line)
                    
                    description_text = ' '.join(clean_lines)
                    # Clean up extra whitespace
                    description_text = re.sub(r'\s+', ' ', description_text).strip()
                    
                    return description_text
            
            return ""
            
        except Exception as e:
            if IS_DEBUG_APP:
                print(f"DEBUG: Error extracting abstract text: {e}")
            return ""
    
    def add_fallback_talk_info(self, content: List[str], talk: Dict, talk_id: str = None):
        """Add fallback information when LaTeX file is not available"""
        title = self.escape_latex(talk.get('title', 'Unknown Title'))
        speaker = self.escape_latex(talk.get('speaker', 'Unknown Speaker'))
        
        content.append(f"\\section{{{title}}}\n\n")
        content.append(f"\\textbf{{Speaker:}} {speaker}\\\\")
        if talk_id:
            content.append(f"\\textbf{{ID:}} {talk_id}\\\\")
        content.append(f"\\textbf{{Relevance Score:}} {talk.get('relevance_score', 0):.1f}\\%\\\\[0.5cm]\n\n")
        
        # Try to get basic info from parsed data
        affiliation = self.escape_latex(talk.get('affiliation', ''))
        if affiliation:
            content.append(f"\\textbf{{Affiliation:}} {affiliation}\\\\")
        
        email = talk.get('email', '')
        if email:
            clean_email = self.escape_latex(email)
            content.append(f"\\textbf{{Email:}} {clean_email}\\\\")
        
        schedule_info = talk.get('schedule_info', {})
        time = schedule_info.get('time', 'TBD')
        room = schedule_info.get('room', 'TBD')
        content.append(f"\\textbf{{Time:}} {time}\\\\")
        content.append(f"\\textbf{{Room:}} {room}\\\\[0.5cm]")
        
        content.append("\\textit{Abstract LaTeX file not available.}\n\n")
        content.append("\\newpage\n\n")
    
    def escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        if not text:
            return ""
        
        # Handle encoding issues first
        try:
            # Try to ensure we have clean UTF-8
            text = text.encode('utf-8', errors='replace').decode('utf-8')
        except:
            # If that fails, replace non-ASCII characters
            text = ''.join(char if ord(char) < 128 else '?' for char in text)
        
        # Handle problematic Unicode characters that cause LaTeX issues
        problematic_chars = {
            '"u': 'u',  # Remove problematic quote-u sequence
            '"{u}': 'u',  # Remove problematic quote-{u} sequence
            '\\"u': 'u',  # Remove escaped quote-u
            'ü': '{"u}',  # Proper LaTeX umlaut - braces will be escaped later
            'ö': '{"o}',  # Proper LaTeX umlaut - braces will be escaped later
            'ä': '{"a}',  # Proper LaTeX umlaut - braces will be escaped later
            'Ü': '{"U}',  # Proper LaTeX umlaut - braces will be escaped later
            'Ö': '{"O}',  # Proper LaTeX umlaut - braces will be escaped later
            'Ä': '{"A}',  # Proper LaTeX umlaut - braces will be escaped later
            'ß': '{\\ss}',   # German sharp s
            '–': '--',       # En dash
            '—': '---',      # Em dash
            ''': "'",        # Smart quote
            ''': "'",        # Smart quote
            '"': '"',        # Smart quote
            '"': '"',        # Smart quote
        }
        
        for char, replacement in problematic_chars.items():
            text = text.replace(char, replacement)
        
        # Protect existing LaTeX math mode before escaping
        text = self.protect_math_mode(text)
        
        # Replace special characters (but preserve protected math)
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '#': r'\#',
            '^': r'\textasciicircum{}',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}'
        }
        
        # Handle backslashes more carefully - avoid incomplete \textbackslash{} at end
        # Protect LaTeX commands that we just created (like {\"u}, {\"o}, etc.)
        # First, replace standalone backslashes or those followed by spaces/end of string
        text = re.sub(r'\\(?=\s|$)', r'\\textbackslash{}', text)
        # Then replace other backslashes that aren't part of LaTeX commands
        # BUT preserve our umlaut commands like {\"u} and standard LaTeX commands
        text = re.sub(r'\\(?![a-zA-Z"\\])', r'\\textbackslash{}', text)
        
        # Don't escape $ in math mode - it's handled by protect_math_mode
        if not re.search(r'MATHMODE\d+PROTECTED', text):
            replacements['$'] = r'\$'
        
        for char, replacement in replacements.items():
            # Skip characters that are part of protected math expressions
            if not re.search(r'MATHMODE\d+PROTECTED', text):
                text = text.replace(char, replacement)
            else:
                # More careful replacement to avoid breaking protected sections
                parts = re.split(r'(MATHMODE\d+PROTECTED)', text)
                for i in range(0, len(parts), 2):  # Only process non-protected parts
                    parts[i] = parts[i].replace(char, replacement)
                text = ''.join(parts)
        
        # Restore protected math mode
        text = self.restore_math_mode(text)
        
        # Extra safety: ensure no unmatched braces
        text = self.fix_unmatched_braces(text)
        
        return text
    
    def protect_math_mode(self, text: str) -> str:
        """Protect LaTeX math mode expressions from being escaped"""
        if not text:
            return ""
        
        # Store math expressions and replace with placeholders
        self._math_expressions = {}
        counter = 0
        
        # Protect inline math $...$
        def replace_inline_math(match):
            nonlocal counter
            placeholder = f"MATHMODE{counter}PROTECTED"
            self._math_expressions[placeholder] = match.group(0)
            counter += 1
            return placeholder
        
        # Protect display math $$...$$
        text = re.sub(r'\$\$([^$]+)\$\$', replace_inline_math, text)
        
        # Protect inline math $...$
        text = re.sub(r'\$([^$]+)\$', replace_inline_math, text)
        
        # Protect \begin{equation}...\end{equation}
        text = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', replace_inline_math, text, flags=re.DOTALL)
        
        # Protect \begin{align}...\end{align}
        text = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', replace_inline_math, text, flags=re.DOTALL)
        
        return text
    
    def restore_math_mode(self, text: str) -> str:
        """Restore protected LaTeX math mode expressions"""
        if not hasattr(self, '_math_expressions'):
            return text
        
        for placeholder, original in self._math_expressions.items():
            text = text.replace(placeholder, original)
        
        # Clear the stored expressions
        self._math_expressions = {}
        
        return text
    
    def make_safe_section_title(self, title: str) -> str:
        """Make a section title safe for LaTeX by handling length and problematic content"""
        if not title:
            return "Untitled"
        
        # First, apply standard escaping but preserve math mode
        safe_title = self.escape_latex(title)
        
        # Check if title is too long - LaTeX sections have limits
        if len(safe_title) > 80:
            # Find a good breaking point
            words = safe_title.split()
            truncated = []
            length = 0
            
            for word in words:
                if length + len(word) + 1 > 77:  # Leave room for "..."
                    break
                truncated.append(word)
                length += len(word) + 1
            
            if truncated:
                safe_title = ' '.join(truncated) + '...'
            else:
                safe_title = safe_title[:77] + '...'
        
        # Extra validation for mathematical titles
        if re.search(r'[A-Za-z]+-[A-Za-z]+', safe_title):
            # Handle hyphenated mathematical terms that might break LaTeX
            safe_title = re.sub(r'([A-Za-z]+)-([A-Za-z]+)', r'\1--\2', safe_title)
        
        # Ensure no line breaks in section titles
        safe_title = re.sub(r'\s+', ' ', safe_title).strip()
        
        return safe_title
    
    def fix_unmatched_braces(self, text: str) -> str:
        """Fix unmatched braces that can cause LaTeX errors"""
        if not text:
            return ""
        
        # Don't process protected math expressions
        if re.search(r'MATHMODE\d+PROTECTED', text):
            return text
        
        # Count braces to detect imbalances (ignore escaped braces)
        open_braces = len([m for m in re.finditer(r'(?<!\\)\{', text)])
        close_braces = len([m for m in re.finditer(r'(?<!\\)\}', text)])
        
        # If severely imbalanced, escape all remaining braces for safety
        if abs(open_braces - close_braces) > 5:
            # Replace any remaining unescaped braces
            text = re.sub(r'(?<!\\)\{', r'\\{', text)
            text = re.sub(r'(?<!\\)\}', r'\\}', text)
        elif open_braces != close_braces:
            # Minor imbalance - try to fix by adding missing braces
            if open_braces > close_braces:
                # Add missing closing braces
                text += '}' * (open_braces - close_braces)
            else:
                # Add missing opening braces at the start
                text = '{' * (close_braces - open_braces) + text
        
        return text
    
    def escape_latex_clean(self, text: str) -> str:
        """Escape special LaTeX characters (excluding backslashes which are already cleaned)"""
        if not text:
            return ""
        
        # Handle encoding issues first
        try:
            # Try to ensure we have clean UTF-8
            text = text.encode('utf-8', errors='replace').decode('utf-8')
        except:
            # If that fails, replace non-ASCII characters
            text = ''.join(char if ord(char) < 128 else '?' for char in text)
        
        # Handle problematic Unicode characters that cause LaTeX issues
        problematic_chars = {
            '"u': 'u',  # Remove problematic quote-u sequence
            '"{u}': 'u',  # Remove problematic quote-{u} sequence
            '\\"u': 'u',  # Remove escaped quote-u
            'ü': '{"u}',  # Proper LaTeX umlaut - braces will be escaped later
            'ö': '{"o}',  # Proper LaTeX umlaut - braces will be escaped later
            'ä': '{"a}',  # Proper LaTeX umlaut - braces will be escaped later
            'Ü': '{"U}',  # Proper LaTeX umlaut - braces will be escaped later
            'Ö': '{"O}',  # Proper LaTeX umlaut - braces will be escaped later
            'Ä': '{"A}',  # Proper LaTeX umlaut - braces will be escaped later
            'ß': '{\\ss}',   # German sharp s
            '–': '--',       # En dash
            '—': '---',      # Em dash
            ''': "'",        # Smart quote
            ''': "'",        # Smart quote
            '"': '"',        # Smart quote
            '"': '"',        # Smart quote
        }
        
        for char, replacement in problematic_chars.items():
            text = text.replace(char, replacement)
        
        # Protect existing LaTeX math mode before escaping
        text = self.protect_math_mode(text)
        
        # Replace special characters (excluding backslash since we cleaned it already)
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '#': r'\#',
            '^': r'\textasciicircum{}',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}'
            # Note: No backslash escaping since we cleaned them in clean_abstract_text
        }
        
        # Don't escape $ in math mode - it's handled by protect_math_mode
        if not re.search(r'MATHMODE\d+PROTECTED', text):
            replacements['$'] = r'\$'
        
        for char, replacement in replacements.items():
            # Skip characters that are part of protected math expressions
            if not re.search(r'MATHMODE\d+PROTECTED', text):
                text = text.replace(char, replacement)
            else:
                # More careful replacement to avoid breaking protected sections
                parts = re.split(r'(MATHMODE\d+PROTECTED)', text)
                for i in range(0, len(parts), 2):  # Only process non-protected parts
                    parts[i] = parts[i].replace(char, replacement)
                text = ''.join(parts)
        
        # Restore protected math mode
        text = self.restore_math_mode(text)
        
        # Extra safety: ensure no unmatched braces
        text = self.fix_unmatched_braces(text)
        
        return text
    
    def clean_abstract_text(self, text: str) -> str:
        """Clean and format abstract text for LaTeX"""
        if not text:
            return ""
        
        # Don't clean if text is very short (likely just a title)
        if len(text.strip()) < 20:
            return self.escape_latex(text.strip())
        
        # First, handle encoding issues by normalizing the text
        try:
            # Try to encode and decode to catch problematic characters
            text = text.encode('utf-8', errors='replace').decode('utf-8')
        except:
            # If that fails, just replace problematic characters
            text = ''.join(char if ord(char) < 128 else '?' for char in text)
        
        # Pre-validate: check for severely malformed content
        if self.is_malformed_content(text):
            return "\\textit{Abstract content appears malformed and cannot be safely processed.}"
        
        # Extract structured information from curly braces first
        structured_info = self.extract_structured_info(text)
        
        # If we found structured info, format it nicely
        if structured_info['has_structure']:
            formatted_text = ""
            
            # Add speaker info if available
            if structured_info['speaker']:
                formatted_text += f"\\textbf{{Speaker:}} {self.escape_latex_clean(structured_info['speaker'])}\\\\[0.2cm]\n"
            
            # Add organization if available
            if structured_info['organization']:
                formatted_text += f"\\textbf{{Organization:}} {self.escape_latex_clean(structured_info['organization'])}\\\\[0.2cm]\n"
            
            # Add email if available
            if structured_info['email']:
                formatted_text += f"\\textbf{{Email:}} {self.escape_latex_clean(structured_info['email'])}\\\\[0.2cm]\n"
            
            # Add special session if available
            if structured_info['special_session']:
                formatted_text += f"\\textbf{{Special Session:}} {self.escape_latex_clean(structured_info['special_session'])}\\\\[0.2cm]\n"
            
            # Add abstract section
            if structured_info['abstract']:
                formatted_text += f"\\textbf{{Abstract:}}\\\\[0.2cm]\n{self.escape_latex_clean(structured_info['abstract'])}"
            
            return formatted_text
        
        # Fallback to general cleaning if no structure found
        return self.general_text_clean(text)
    
    def is_malformed_content(self, text: str) -> bool:
        """Check if content is severely malformed and might cause LaTeX errors"""
        if not text:
            return False
        
        # Check for excessive brace imbalance
        open_braces = text.count('{')
        close_braces = text.count('}')
        if abs(open_braces - close_braces) > 10:  # Allow some imbalance, but not excessive
            return True
        
        # Check for excessive backslashes (might indicate malformed LaTeX)
        if text.count('\\') > len(text) / 10:  # More than 10% backslashes is suspicious
            return True
        
        # Check for very long lines without spaces (might break LaTeX)
        lines = text.split('\n')
        for line in lines:
            if len(line) > 500 and ' ' not in line:
                return True
        
        return False
    
    def extract_structured_info(self, text: str) -> Dict:
        """Extract structured information from LaTeX talk format"""
        info = {
            'has_structure': False,
            'speaker': '',
            'organization': '',
            'email': '',
            'special_session': '',
            'abstract': ''
        }
        
        # Look for pattern with curly braces containing structured data
        # Pattern: {Speaker} {Organization} {Email} {CoAuthors?} {SpecialSession} AbstractText
        brace_pattern = r'\{([^}]*)\}'
        braces = re.findall(brace_pattern, text)
        
        if len(braces) >= 3:
            info['has_structure'] = True
            
            # Extract from typical positions
            if len(braces) >= 1 and braces[0].strip():
                info['speaker'] = braces[0].strip()
            
            if len(braces) >= 2 and braces[1].strip():
                info['organization'] = braces[1].strip()
            
            if len(braces) >= 3 and braces[2].strip():
                info['email'] = braces[2].strip()
            
            # Look for special session (usually the last meaningful brace)
            if len(braces) >= 5 and braces[4].strip():
                info['special_session'] = braces[4].strip()
            
            # Extract abstract text (everything after the last brace)
            last_brace_pos = text.rfind('}')
            if last_brace_pos != -1:
                abstract_text = text[last_brace_pos + 1:].strip()
                # Clean the abstract text
                abstract_text = self.clean_abstract_content(abstract_text)
                info['abstract'] = abstract_text
        
        return info
    
    def clean_abstract_content(self, text: str) -> str:
        """Clean just the abstract content part"""
        if not text:
            return ""
        
        # Remove LaTeX comments (% to end of line)
        text = re.sub(r'%.*?(?=\n|$)', '', text, flags=re.MULTILINE)
        
        # Remove numbered field metadata patterns like [1], [2], [3], etc.
        text = re.sub(r'\[\d+\][^{]*{[^}]*}', '', text)
        
        # Clean LaTeX commands more systematically
        # First, handle multiple backslashes (line breaks)
        text = re.sub(r'\\\\+', ' ', text)
        
        # Remove LaTeX begin/end environments
        text = re.sub(r'\\begin\{[^}]*\}', '', text)
        text = re.sub(r'\\end\{[^}]*\}', '', text)
        
        # Remove LaTeX commands but preserve their content
        text = re.sub(r'\\[a-zA-Z]+\*?\s*\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)
        
        # Clean up remaining single backslashes
        text = re.sub(r'\\', ' ', text)
        
        # Remove template instructions
        instruction_patterns = [
            r'Your abstract goes here.*?macros\.',
            r'Please do not use.*?macros\.',
            r'Insert the title.*?session\.',
            r'Leave this field empty.*?talks\.',
            r'If you would like.*?below\.',
            r'Please do not use.*?files\.',
            r'APA reference style.*?recommended\.',
            r'Equations may be used.*?book\.'
        ]
        
        for pattern in instruction_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_latex_for_display(self, text: str) -> str:
        """Clean LaTeX content and convert it to readable text for display in PDFs"""
        if not text:
            return ""
        
        # Remove LaTeX comments (% to end of line)
        text = re.sub(r'%.*?(?=\n|$)', '', text, flags=re.MULTILINE)
        
        # Convert LaTeX formatting to readable text
        # Handle spacing commands
        text = re.sub(r'\\medskip', '\n\n', text)
        text = re.sub(r'\\bigskip', '\n\n\n', text)
        text = re.sub(r'\\smallskip', '\n', text)
        text = re.sub(r'\\vspace\{[^}]*\}', '\n', text)
        
        # Handle lists - convert enumerate to numbered lists
        enumerate_pattern = r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}'
        def convert_enumerate(match):
            content = match.group(1)
            items = re.findall(r'\\item\s+(.*?)(?=\\item|$)', content, re.DOTALL)
            result = ""
            for i, item in enumerate(items, 1):
                item = item.strip()
                if item:
                    result += f"{i}. {item}\n"
            return result
        
        text = re.sub(enumerate_pattern, convert_enumerate, text, flags=re.DOTALL)
        
        # Handle itemize lists
        itemize_pattern = r'\\begin\{itemize\}(.*?)\\end\{itemize\}'
        def convert_itemize(match):
            content = match.group(1)
            items = re.findall(r'\\item\s+(.*?)(?=\\item|$)', content, re.DOTALL)
            result = ""
            for item in items:
                item = item.strip()
                if item:
                    result += f"• {item}\n"
            return result
        
        text = re.sub(itemize_pattern, convert_itemize, text, flags=re.DOTALL)
        
        # Handle line breaks
        text = re.sub(r'\\\\+', '\n', text)
        text = re.sub(r'\\newline', '\n', text)
        
        # Remove remaining LaTeX begin/end environments
        text = re.sub(r'\\begin\{[^}]*\}', '', text)
        text = re.sub(r'\\end\{[^}]*\}', '', text)
        
        # Handle text formatting commands by keeping content
        text = re.sub(r'\\textbf\{([^}]*)\}', r'**\1**', text)  # Bold
        text = re.sub(r'\\textit\{([^}]*)\}', r'*\1*', text)    # Italic
        text = re.sub(r'\\emph\{([^}]*)\}', r'*\1*', text)      # Emphasis
        text = re.sub(r'\\texttt\{([^}]*)\}', r'`\1`', text)    # Monospace
        
        # Remove other LaTeX commands but preserve their content
        text = re.sub(r'\\[a-zA-Z]+\*?\s*\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)
        
        # Clean up remaining backslashes
        text = re.sub(r'\\', ' ', text)
        
        # Clean up excessive whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 newlines
        text = re.sub(r'[ \t]+', ' ', text)             # Multiple spaces to single
        text = re.sub(r' *\n *', '\n', text)            # Spaces around newlines
        
        return text.strip()
    
    def general_text_clean(self, text: str) -> str:
        """General text cleaning for non-structured content"""
        # Remove LaTeX comments (% to end of line)
        text = re.sub(r'%.*?(?=\n|$)', '', text, flags=re.MULTILINE)
        
        # Remove numbered field metadata patterns
        text = re.sub(r'\[\d+\][^{]*{[^}]*}', '', text)
        
        # Clean LaTeX commands
        text = re.sub(r'\\\\+', ' ', text)
        text = re.sub(r'\\begin\{[^}]*\}', '', text)
        text = re.sub(r'\\end\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+\*?\s*\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)
        text = re.sub(r'\\', ' ', text)
        
        # Remove template instructions
        instruction_patterns = [
            r'Your abstract goes here.*?macros\.',
            r'Please do not use.*?macros\.',
            r'Insert the title.*?session\.',
            r'Leave this field empty.*?talks\.',
            r'If you would like.*?below\.',
            r'Please do not use.*?files\.',
            r'APA reference style.*?recommended\.',
            r'Equations may be used.*?book\.'
        ]
        
        for pattern in instruction_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Don't escape if the result is still very short
        if len(text) < 20:
            return "\\textit{Abstract content not available or too brief.}"
        
        # Escape LaTeX special characters
        text = self.escape_latex_clean(text)
        
        return text
    
    def compile_latex_to_pdf(self, tex_file_path: str, output_file: Path):
        """Compile LaTeX to PDF"""
        try:
            # Change to the directory containing the tex file
            tex_dir = Path(tex_file_path).parent
            tex_filename = Path(tex_file_path).name
            
            # Run pdflatex twice to resolve references
            for i in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', tex_filename],
                    cwd=tex_dir,
                    capture_output=True,
                    text=True,
                    encoding='latin1',  # Use latin1 to handle problematic characters
                    errors='replace'    # Replace problematic characters instead of failing
                )
                
                # Check if PDF was actually generated (more reliable than return code)
                pdf_source = tex_dir / tex_filename.replace('.tex', '.pdf')
                if not pdf_source.exists():
                    error_msg = f"LaTeX compilation failed (attempt {i+1}) - PDF not generated"
                    if result.stderr:
                        error_msg += f": {result.stderr}"
                    if result.stdout:
                        # Look for actual error indicators in output
                        if "Error" in result.stdout or "Fatal error" in result.stdout:
                            error_msg += f"\nErrors found: {result.stdout[-500:]}"
                    raise Exception(error_msg)
            
            # Move the PDF to the desired location
            pdf_source = tex_dir / tex_filename.replace('.tex', '.pdf')
            if pdf_source.exists():
                shutil.move(str(pdf_source), str(output_file))
            else:
                # List files in directory for debugging
                files_in_dir = list(tex_dir.glob('*'))
                raise Exception(f"PDF file was not generated. Files in {tex_dir}: {[f.name for f in files_in_dir]}")
                
        except FileNotFoundError as e:
            raise Exception(f"pdflatex not found. Please install LaTeX (e.g., MacTeX on macOS). Error: {str(e)}")
        except Exception as e:
            if "LaTeX compilation failed" in str(e) or "PDF file was not generated" in str(e) or "pdflatex not found" in str(e):
                raise  # Re-raise our custom exceptions
            else:
                raise Exception(f"PDF compilation failed with unexpected error: {str(e)}")
    
    def run(self):
        """Run the Streamlit application"""
        # Header
        st.title("MCM 2025: Personalized Program Book Generator")
        #st.subheader("AI  Program Generator")
        
        # AI Status in sidebar
        #st.sidebar.markdown("---")
        #st.sidebar.subheader("AI Assistant")
        
        # Model selection
        available_models = ["qwen3", "gemma3n"]
        current_model = st.session_state.get('selected_model', "qwen3")
        
        selected_model = st.sidebar.selectbox(
            "**Choose AI Model:**",
            available_models,
            index=available_models.index(current_model) if current_model in available_models else 0,
            help="Select an AI model"
        )
        
        # Update model if changed
        if selected_model != st.session_state.get('selected_model'):
            st.session_state.selected_model = selected_model
            # Reinitialize LLM with new model
            with st.spinner(f"Switching to {selected_model}..."):
                st.session_state.llm_assistant = LLMAssistant(model_name=selected_model)
                self.llm = st.session_state.llm_assistant
            st.rerun()
        
        if self.llm.available:
            model_display = self.llm.model_name
            if model_display.startswith("qwen"):
                if "3" in model_display:
                    model_display = "Qwen 3"
            elif model_display.startswith("gemma"):
                if "3" in model_display:
                    model_display = "Gemma 3n"
            self.ai_model_name = selected_model
            st.sidebar.success(f"✅ Active")# ({selected_model})")
            
            # Add model info
            #st.sidebar.info(f"📡 Current: {selected_model}")
        else:
            st.sidebar.warning("⚠️ AI Service Offline")
            st.sidebar.code("ollama serve")
            st.sidebar.info("Start Ollama service to enable AI features")
            
            # Show more detailed debug info
            with st.sidebar.expander("Debug Info"):
                st.write("**Issue:** Cannot connect to Ollama or models not found")
                st.write("**Expected models:** qwen3, gemma3n")
                st.write("**Try these commands:**")
                st.code("ollama pull qwen3\nollama pull gemma3n")
                st.write("**Check service:**")
                st.code("ollama list")
        
        # Data loading status
        #st.sidebar.markdown("---")
        #st.sidebar.subheader("📊 Data Status")
        #if st.session_state.data_loaded:
        #    st.sidebar.success(f"✅ {len(st.session_state.abstracts_data)} abstracts loaded")
        #else:
        #    st.sidebar.warning("⚠️ Data not loaded")

        # Support the Developer section
        st.sidebar.markdown("---")
        st.sidebar.subheader("☕ Support the Developer")
        st.sidebar.markdown(""" Made with ❤️ for the MCM 2025 community.
        If this tool has helped you create a useful conference schedule, consider buying me a coffee! ☕
        Thank you for your support! 🙏
        """)
        
        st.sidebar.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <a href='https://gofund.me/ed7c87a0' target='_blank' 
               style='background-color: #00D2FF; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 8px; font-weight: bold;
                      display: inline-block; margin: 5px;'>
                ☕ Buy Me a Coffee
            </a>
            <br>
            <a href='https://www.linkedin.com/in/sou-cheng-choi-7682b65/' target='_blank' 
               style='background-color: #0077B5; color: white; padding: 10px 20px; 
                      text-decoration: none; border-radius: 8px; font-weight: bold;
                      display: inline-block; margin: 5px;'>
                💼 LinkedIn Profile
            </a>
        </div>
        """, unsafe_allow_html=True)
        
       #st.sidebar.markdown("*Thank you for your support! 🙏*")
        
        # Page content - Direct to streamlined experience
        self.streamlined_page()
        
        # Footer with donation link
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px 0;'>
            <small>
                Made with ❤️ for the MCM 2025 community | 
                <a href='https://gofund.me/ed7c87a0' target='_blank' style='color: #FF6B6B;'>☕ Buy me a coffee</a> 
                if this app helped you!
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    def streamlined_page(self):
        """Streamlined single-page experience: Interests → Find Talks → Generate PDF"""
        #st.subheader("Make your own MCM 2025 schedule!")
     
        
        # Step 1: Research Interests
        st.subheader("Step 1: Enter Your Research Keywords ")
        with st.expander("", expanded=True):
            st.info("🔍 Tips:l ower case letters, singular noun, comma-separated, no other punctuation")
            keywords_text = st.text_area(
                "Keywords",
                value="qmc, computational mathematics, statistics, numerical linear algebra, matrix computations, data science, artificial intelligence, AI, large language model, machine learning, finance and insurance application, quantum computing, algorithm, software, hardware",
                height=80,
                help="Enter terms that describe your research interests (comma-separated)",
                label_visibility="collapsed"
            )
            
        st.subheader("Step 2: Additional Preferences (optional)")
        with st.expander("", expanded=True):
            #st.info("💡 **Tip:** Enter your preferences here. The AI will automatically manage talk limits (10/day, 45 total) and resolve conflicts.")
            preferences = st.text_area(
                "Additional preferences:",
                value="Please do not include talks from 1.30p to 3.30p on Thursday. Include special sessions co-organized by Sou-Cheng Choi.",
                height=80,
                help="Enter your personal constraints: time preferences, preferred speakers/organizers, content focus, etc. System will automatically handle talk limits and conflict resolution.",
                label_visibility="collapsed"
            )

        # Show data loading status
        consolidated_file = self.base_path.parent / "MCM_ProgramBook_TEX" / "MCM2025_consolidated.tex"
        if not consolidated_file.exists():
            #st.info(f"🚀 **Enhanced Performance**: Using consolidated LaTeX file ({consolidated_file.stat().st_size:,} bytes) #for faster loading!")
        #else:
            st.info("📁 **Loading Status**: Using individual LaTeX files (consolidated file not available)")

        
        # Step 2: Generate Personalized Schedule
        if st.button("Generate Personalized Schedule", type="primary", use_container_width=True):
            keywords = [k.strip().lower() for k in keywords_text.split(',') if k.strip()]
            
            if not keywords:
                st.error("Please enter at least some keywords.")
                return
            
            # Create user interests dictionary
            user_interests = {
                'keywords': keywords,
                'areas': keywords,  # Use keywords as areas for simplicity
                'experience': 'Intermediate',
                'preferences': preferences
            }
            
            # Store user interests in session state for AI selection
            st.session_state.user_interests = user_interests
            
            with st.spinner("🤖 AI is analyzing parallel sessions and creating your personalized schedule... It may take a minute..."):
                stime = time.time()
                # Generate personalized schedule
                personalized_schedule = self.generate_personalized_schedule(user_interests)
                
                if personalized_schedule:
                    # Generate abstracts for selected sessions
                    selected_sessions = st.session_state.get('selected_parallel_sessions', [])
                    personalized_abstracts = self.generate_personalized_abstracts(selected_sessions)
                    
                    # Create complete LaTeX document
                    full_latex = self.create_personalized_latex_document(
                        personalized_schedule, 
                        personalized_abstracts, 
                        user_interests
                    )
                    
                    # Generate PDF
                    if full_latex:
                        self.compile_and_download_personalized_pdf(full_latex, user_interests)
                    else:
                        st.error("Failed to generate LaTeX document")
                else:
                    st.error("Failed to generate personalized schedule")
                elapsed_time = time.time() - stime

            st.info(f"{elapsed_time = :.0f} seconds")
    
    def generate_streamlined_pdf(self, relevant_talks, include_abstracts=True, include_ai_insights=False, schedule_only=False):
        """Streamlined PDF generation with automatic conflict resolution and download"""
        st.info(f"🔄 Starting PDF generation...")
        
        # Start timing overall PDF generation
        pdf_start_time = time.time()
        
        try:
            # Start timing conflict resolution
            conflict_start_time = time.time()
            
            with st.spinner("🤖 AI is creating your conflict-free personalized schedule..."):
                
                # Filter for talks with substantial content (abstracts)
                talks_with_abstracts = []
                session_talks = []
                
                # Enhanced abstract content extraction with fallback mechanism
                if IS_DEBUG_APP:
                    st.write("🔍 **Debug: Analyzing and enhancing talk sources:**")
                source_counts = {'abstract': 0, 'schedule': 0, 'unknown': 0}
                
                for talk in relevant_talks:
                    abstract = talk.get('abstract', '')
                    source = talk.get('source', '')
                    title = talk.get('title', 'Unknown Title')
                    talk_id = talk.get('id', '')
                    
                    # Count sources
                    if source in source_counts:
                        source_counts[source] += 1
                    else:
                        source_counts['unknown'] += 1
                    
                    # For abstract source talks with missing content, try to re-extract from LaTeX file
                    if source == 'abstract' and (not abstract or len(abstract.strip()) <= 20) and talk_id:
                        try:
                            # Try to find and re-extract from LaTeX file
                            abstract_file_path = self.abstracts_path / f"{talk_id}.tex"
                            if abstract_file_path.exists():
                                full_content = self.extract_talk_content_from_file(str(abstract_file_path))
                                if full_content:
                                    # Extract just the abstract text portion
                                    enhanced_abstract = self.extract_abstract_text_from_latex(full_content)
                                    if enhanced_abstract and len(enhanced_abstract.strip()) > 20:
                                        talk['abstract'] = enhanced_abstract
                                        abstract = enhanced_abstract
                                        if len(talks_with_abstracts) + len(session_talks) < 10:
                                            st.write(f"  🔧 **Enhanced abstract from LaTeX file** (new length: {len(enhanced_abstract)})")
                        except Exception as e:
                            # Silent fallback - don't show error to user
                            pass
                    
                    # Debug output for first few talks
                    if len(talks_with_abstracts) + len(session_talks) < 10:
                        st.write(f"- **{title[:50]}...** | Source: `{source}` | Abstract length: `{len(abstract)}` chars")
                    
                    # More inclusive approach - include if:
                    # 1. From abstract files with substantial content, OR
                    # 2. Plenary talks (regardless of source), OR 
                    # 3. High-value sessions with some content
                    is_plenary = talk.get('type') == 'plenary' or 'plenary' in title.lower()
                    is_substantial_abstract = source == 'abstract' and abstract and len(abstract.strip()) > 20
                    is_valuable_session = (talk.get('type') in ['session', 'special_session'] and 
                                         abstract and len(abstract.strip()) > 10)
                    
                    if is_substantial_abstract or is_plenary or is_valuable_session:
                        talks_with_abstracts.append(talk)
                        if IS_DEBUG_APP and (len(talks_with_abstracts) + len(session_talks) < 10):  
                            if is_plenary:
                                st.write(f"  ✅ **Added to abstracts group** (PLENARY - source: {source}, length: {len(abstract)})")
                            elif is_substantial_abstract:
                                st.write(f"  ✅ **Added to abstracts group** (ABSTRACT - length: {len(abstract)})")
                            elif is_valuable_session:
                                st.write(f"  ✅ **Added to abstracts group** (SESSION - length: {len(abstract)})")
                    else:
                        session_talks.append(talk)
                        if len(talks_with_abstracts) + len(session_talks) < 10:
                            if not abstract:
                                st.write(f"  ❌ **No abstract content found**")
                            elif abstract and len(abstract.strip()) <= 20:
                                st.write(f"  ❌ **Abstract too short** ({len(abstract)} chars): `{abstract[:100]}...`")
                            elif source != 'abstract' and not is_plenary:
                                st.write(f"  ❌ **Wrong source type** ({source}) - not plenary")
                
                st.write(f"📊 **Source distribution:** Abstract files: {source_counts['abstract']}, Schedule: {source_counts['schedule']}, Unknown: {source_counts['unknown']}")
                st.write(f"📚 **Filtering results:** {len(talks_with_abstracts)} talks with substantial abstracts, {len(session_talks)} other talks")
                
                # Prioritize talks with abstracts first, then add session talks
                all_talks_for_conflict_resolution = talks_with_abstracts + session_talks
                
                st.write(f"📊 **Final talk order:** {len(talks_with_abstracts)} abstract talks first, then {len(session_talks)} session talks")
                
                # Convert to format expected by conflict resolution
                talks_with_scores = []
                for talk in all_talks_for_conflict_resolution:
                    schedule_info = talk.get('schedule_info', {})
                    time_slot = schedule_info.get('time', '')
                    talks_with_scores.append({
                        'talk': talk,
                        'time': time_slot,
                        'relevance_score': talk.get('relevance_score', 0)
                    })
                
                # Sort by relevance score before conflict resolution
                talks_with_scores.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                
                # Apply automatic conflict resolution with daily and total limits
                conflict_free_talks = self.select_talks_without_conflicts(talks_with_scores)
                
                # Show daily distribution of selected talks
                if conflict_free_talks:
                    daily_counts = {}
                    day_keywords = {
                        'monday': 'Monday', 'mon': 'Monday',
                        'tuesday': 'Tuesday', 'tue': 'Tuesday',
                        'wednesday': 'Wednesday', 'wed': 'Wednesday',
                        'thursday': 'Thursday', 'thu': 'Thursday',
                        'friday': 'Friday', 'fri': 'Friday'
                    }
                    
                    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                        daily_counts[day] = 0
                    
                    for item in conflict_free_talks:
                        time_slot = item.get('time', '')
                        if time_slot:
                            time_lower = time_slot.lower()
                            day = 'Unknown'
                            for keyword, day_name in day_keywords.items():
                                if keyword in time_lower:
                                    day = day_name
                                    break
                            if day != 'Unknown':
                                daily_counts[day] += 1
                    
                    st.write("📅 **Talk distribution by day (max 10 per day, max 45 total):**")
                    for day, count in daily_counts.items():
                        if count > 0:
                            limit_indicator = "✅" if count <= 10 else "⚠️"
                            st.write(f"  {limit_indicator} **{day}**: {count} talks")
                    
                    total_scheduled = sum(daily_counts.values())
                    total_indicator = "✅" if total_scheduled <= 45 else "⚠️"
                    st.write(f"  {total_indicator} **Total**: {total_scheduled} talks (limit: 45)")
                
                # Fallback: If conflict resolution failed, just take top talks
                if len(conflict_free_talks) == 0:
                    st.info("🔧 Using intelligent talk selection to optimize your schedule...")
                    # Take top 45 talks by relevance score without conflict resolution
                    conflict_free_talks = talks_with_scores[:45]
                
                # Extract just the talk objects
                selected_talks = [item['talk'] for item in conflict_free_talks]
                
                # Debug: Show what talks were selected
                st.write("🎯 **Selected talks summary:**")
                for i, talk in enumerate(selected_talks[:5]):  # Show first 5
                    title = talk.get('title', 'Unknown Title')
                    source = talk.get('source', 'unknown')
                    abstract_len = len(talk.get('abstract', ''))
                    relevance = talk.get('relevance_score', 0)
                    st.write(f"{i+1}. **{title[:40]}...** | Source: `{source}` | Abstract: `{abstract_len}` chars | Score: `{relevance:.1f}`")
                
                if len(selected_talks) > 5:
                    st.write(f"... and {len(selected_talks) - 5} more talks")
                
                if len(selected_talks) == 0:
                    st.error("❌ No talks found matching your criteria!")
                    st.write("Try adjusting your keywords or research areas.")
                    return
                
                # Calculate conflict resolution time
                conflict_resolution_time = time.time() - conflict_start_time
                #st.success(f"✅ Generated conflict-free schedule with {len(selected_talks)} talks!")
                #st.info(f"⏱️ **Conflict Resolution Time:** {conflict_resolution_time:.2f} seconds")

                # Generate filename with AI model name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                base_filename = f"MCM2025_PersonalProgram_{timestamp}"
                
                # Add AI model name to filename if available
                if self.llm.available and hasattr(self.llm, 'model_name'):
                    model_name = self.llm.model_name
                    if model_name:
                        model_tag = model_name.split(":")[0]  # Remove :latest if present
                        filename = f"{base_filename}_{model_tag}.pdf"
                    else:
                        filename = f"{base_filename}.pdf"
                else:
                    filename = f"{base_filename}.pdf"

                # Start timing PDF compilation
                pdf_compile_start_time = time.time()

                # Generate PDF with the selected talks
                with st.spinner("📄 Compiling your personalized program book..."):
                    output_file = self.generate_pdf(
                        selected_talks,
                        filename,
                        include_abstracts,
                        True,  # include_schedule
                        False  # include_conflicts (since we resolved them)
                    )

                # Calculate PDF compilation time
                pdf_compile_time = time.time() - pdf_compile_start_time
                total_time = time.time() - pdf_start_time

                # Success celebration
                st.balloons()
                #st.success("🎉 Your conflict-free personalized conference program is ready!")
                
                # Show detailed timing information
                #st.info(f"⏱️ **PDF Compilation Time:** {pdf_compile_time:.2f} seconds")
                #st.info(f"⏱️ **Total PDF Compilation and Processing Time:** {total_time:.1f} seconds")
                
                # Show LaTeX source information
                tex_filename = filename.replace('.pdf', '.tex')

                # Automatic download
                with open(output_file, 'rb') as pdf_file:
                    pdf_data = pdf_file.read()
                    
                    # Create download button that auto-triggers
                    st.download_button(
                        label="📥 Download Your Program (Click to Save)",
                        data=pdf_data,
                        file_name=filename,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                        key=f"download_{timestamp}"  # Unique key to avoid conflicts
                    )

                # Show program summary
                st.markdown(f"""
                <div style='background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1); 
                           padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;'>
                    <h3 style='color: white; margin: 0;'>� Your Conflict-Free Program Summary</h3>
                    <p style='color: white; margin: 10px 0;'>
                        ✅ {len(selected_talks)} carefully selected talks<br>
                        🚫 Zero scheduling conflicts<br>
                        📚 Full abstracts included<br>
                        🤖 AI-optimized for your interests<br>
                        ⏱️ Generated in {total_time:.1f} seconds
                    </p>
                    <p style='color: white; margin: 5px 0; font-size: 0.9em;'>
                        Conflict Resolution: {conflict_resolution_time:.1f}s | PDF Compilation: {pdf_compile_time:.1f}s
                    </p>
                    <a href='https://gofund.me/ed7c87a0' target='_blank' 
                       style='background: white; color: #333; padding: 10px 20px; 
                              text-decoration: none; border-radius: 5px; font-weight: bold;
                              display: inline-block; margin-top: 10px;'>
                        ☕ Buy Me a Coffee!
                    </a>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            # Calculate time up to error
            error_time = time.time() - pdf_start_time
            
            st.error(f"❌ Error generating PDF: {str(e)}")
            st.info(f"⏱️ **Time before error:** {error_time:.2f} seconds")
            
            # More detailed error reporting
            import traceback
            error_details = traceback.format_exc()
            st.error(f"Full error details: {error_details}")
            
            # Debug information
            st.write(f"Debug info:")
            st.write(f"- Number of relevant talks: {len(relevant_talks)}")
            st.write(f"- Selected talks count: {len(selected_talks) if 'selected_talks' in locals() else 'Not created'}")
            st.write(f"- Output path exists: {self.output_path.exists()}")
            st.write(f"- Output path: {self.output_path}")
            
            st.info("Please check that:")
            st.info("- LaTeX is installed (try: brew install --cask mactex)")
            st.info("- Conference data files are properly loaded")
            st.info("- You have write permissions in the output directory")
    
    def extract_talk_content_from_data(self, abstract_data: Dict) -> str:
        """Extract LaTeX talk content from comprehensive abstract data"""
        if not abstract_data.get('content'):
            return None
        
        content = abstract_data['content']
        
        # Find the \begin{talk} and \end{talk} boundaries in the content
        start_match = re.search(r'\\begin\{talk\}', content)
        end_match = re.search(r'\\end\{talk\}', content)
        
        if start_match and end_match:
            # Extract the talk content including the begin/end tags
            talk_content = content[start_match.start():end_match.end()]
            return talk_content
        
        return None
    
    def extract_session_content_from_data(self, session_data: Dict) -> str:
        """Extract LaTeX session content from comprehensive session data"""
        if not session_data.get('content'):
            return None
        
        content = session_data['content']
        
        # Find the \begin{session} and \end{session} boundaries in the content
        start_match = re.search(r'\\begin\{session\}', content)
        end_match = re.search(r'\\end\{session\}', content)
        
        if start_match and end_match:
            # Extract the session content including the begin/end tags
            session_content = content[start_match.start():end_match.end()]
            return session_content
        
        return None
    
    def add_comprehensive_talk_info(self, content: List[str], abstract_data: Dict, talk_number: int):
        """Add formatted talk information from comprehensive abstract data"""
        title = self.escape_latex(abstract_data.get('title', 'Unknown Title'))
        speaker = self.escape_latex(abstract_data.get('speaker', 'Unknown Speaker'))
        talk_id = abstract_data.get('id', 'Unknown')
        
        content.append(f"\\section{{{title}}}\n\n")
        content.append(f"% DEBUG: Talk ID = {talk_id}\n")
        content.append(f"% DEBUG: Comprehensive time_slot = {abstract_data.get('time_slot', 'None')}\n")
        content.append(f"% DEBUG: Schedule info time = {abstract_data.get('schedule_info', {}).get('time', 'None')}\n")
        
        content.append(f"\\textbf{{Speaker:}} {speaker}\\\\")
        
        # Add comprehensive information
        if abstract_data.get('affiliation'):
            affiliation = self.escape_latex(abstract_data['affiliation'])
            content.append(f"\\textbf{{Affiliation:}} {affiliation}\\\\")
        
        if abstract_data.get('email'):
            email = self.escape_latex(abstract_data['email'])
            content.append(f"\\textbf{{Email:}} {email}\\\\")
        
        if abstract_data.get('coauthors'):
            coauthors = self.escape_latex(abstract_data['coauthors'])
            content.append(f"\\textbf{{Coauthors:}} {coauthors}\\\\")
        
        if abstract_data.get('special_session'):
            special_session = self.escape_latex(abstract_data['special_session'])
            content.append(f"\\textbf{{Special Session:}} {special_session}\\\\")
        
        # Prioritize comprehensive time_slot over schedule_info time
        time_to_display = ''
        if abstract_data.get('time_slot'):
            time_to_display = abstract_data['time_slot']
            content.append(f"% Using comprehensive time slot\n")
        elif abstract_data.get('schedule_info', {}).get('time'):
            time_to_display = abstract_data['schedule_info']['time']
            content.append(f"% Using CSV schedule time (fallback)\n")
        
        if time_to_display and time_to_display != 'TBD':
            time_escaped = self.escape_latex(time_to_display)
            content.append(f"\\textbf{{Time:}} {time_escaped}\\\\")
        else:
            content.append(f"\\textbf{{Time:}} TBD\\\\")
        
        content.append(f"\\textbf{{Talk ID:}} {talk_id}\\\\[0.5cm]\n\n")
        
        # Add abstract content if available
        if abstract_data.get('abstract'):
            content.append("\\textbf{Abstract:}\\\\[0.3cm]\n")
            abstract_text = self.escape_latex(abstract_data['abstract'])
            content.append(f"{abstract_text}\n\n")
        else:
            content.append("\\textit{No abstract text available.}\n\n")
        
        content.append("\\newpage\n\n")
    
    def add_session_info_to_content(self, content: List[str], session_data: Dict, session_number: int):
        """Add session information using raw LaTeX input files instead of generating summaries"""
        session_id = session_data.get('id', '')
        title = self.escape_latex(session_data.get('title', f'Session {session_number}'))
        
        # Create section for the session
        content.append(f"\\section{{Session {session_number} – {session_id}: {title}}}\n\n")
        
        # Input the session schedule file from MCM_ProgramBook_TEX
        content.append(f"% Including session schedule file via \\input\n")
        content.append(f"\\input{{{session_id}_sess_talks}}\n\n")
        
        content.append("\n\\newpage\n\n")

    def generate_personalized_schedule(self, user_interests: Dict) -> str:
        """Generate a personalized schedule by selecting the best parallel sessions for each time slot"""
        if not self.llm.available:
            st.warning("AI assistant is not available. Cannot generate personalized schedule.")
            return None
            
        try:
            # Load the main schedule template
            schedule_tex_path = self.base_path.parent / "MCM_ProgramBook_TEX" / "Schedule.tex"
            if not schedule_tex_path.exists():
                st.error(f"Schedule.tex not found at {schedule_tex_path}")
                return None
                
            schedule_content = self.safe_read_file(schedule_tex_path)
            if not schedule_content:
                st.error(f"Failed to read Schedule.tex from {schedule_tex_path}")
                return None
            
            # Parse the schedule to identify time slots with parallel sessions
            time_slots = self.parse_schedule_time_slots(schedule_content)
            
            # Process time slots in order to maintain the original schedule structure
            personalized_schedule_parts = []
            selected_sessions_data = []
            
            for i, time_slot in enumerate(time_slots):
                if time_slot['has_parallel_sessions']:
                    # Load data for all parallel sessions in this time slot
                    parallel_sessions = self.load_parallel_session_data(time_slot)
                    
                    if parallel_sessions:
                        # Extract session IDs for AI selection
                        session_ids = [session['id'] for session in parallel_sessions]
                        day_period = time_slot['label'].split() if time_slot['label'] else ['Unknown', 'Unknown']
                        day = day_period[0] if len(day_period) > 0 else 'Unknown'
                        period = day_period[1] if len(day_period) > 1 else 'Unknown'
                        
                        # DEBUG: Show time slot processing order and session details
                        print(f"\n{'='*60}")
                        #print(f"DEBUG: Processing Time Slot #{i+1}")
                        #print(f"DEBUG: Day/Period: {day} {period}")
                        #print(f"DEBUG: Time Slot Label: '{time_slot['label']}'")
                        #print(f"DEBUG: Available Parallel Session IDs: {session_ids}")
                        #print(f"DEBUG: Total Parallel Sessions: {len(parallel_sessions)}")
                        
                        # Show detailed information for each session that will be sent to AI
                        #print(f"DEBUG: Session Details Being Sent to AI:")
                        for j, session in enumerate(parallel_sessions):
                            print(f"  Session {j+1}: {session['id']}")
                            print(f"    Title: '{session.get('title', 'Unknown')}'")
                            print(f"    Type: '{session.get('type', 'Unknown')}'")
                            print(f"    Room: '{session.get('room', 'Unknown')}'")
                            
                            # Show organizer information
                            organizers = session.get('organizers', [])
                            if organizers and isinstance(organizers, list):
                                org_names = [org.get('name', 'Unknown') for org in organizers]
                                print(f"    Organizers: {org_names}")
                            elif session.get('organizer_from_schedule'):
                                print(f"    Organizer (from schedule): '{session.get('organizer_from_schedule')}'")
                            elif session.get('chair_from_schedule'):
                                print(f"    Chair (from schedule): '{session.get('chair_from_schedule')}'")
                            else:
                                print(f"    Organizers: None found")
                            
                            # Show description if available
                            desc = session.get('description', '')
                            if desc:
                                desc_preview = desc[:100] + "..." if len(desc) > 100 else desc
                                print(f"    Description: '{desc_preview}'")
                            else:
                                print(f"    Description: Not available")
                            
                            # Show talk count
                            talk_count = len(session.get('talks', []))
                            print(f"    Number of Talks: {talk_count}")
                        
                        #print(f"DEBUG: Calling AI to select best session...")
                        
                        # Use AI to select the best session
                        best_session_id = self.ai_select_best_session(
                            session_ids, 
                            day, 
                            period, 
                            []  # selected_talks not needed for this context
                        )
                        
                        # DEBUG: Show AI selection result
                        #print(f"DEBUG: AI Selected Session ID: '{best_session_id}'")
                        
                        # Find the selected session data
                        best_session = None
                        for session in parallel_sessions:
                            if session['id'] == best_session_id:
                                best_session = session
                                break
                        
                        if best_session:
                            #print(f"DEBUG: AI Selection Details:")
                            print(f"  Selected: {best_session['id']} - '{best_session.get('title', 'Unknown')}'")
                            # Show why this might be a good choice
                            organizers = best_session.get('organizers', [])
                            if organizers and isinstance(organizers, list):
                                org_names = [org.get('name', 'Unknown') for org in organizers]
                                print(f"  Organizers: {org_names}")
                            elif best_session.get('organizer_from_schedule'):
                                print(f"  Organizer: {best_session.get('organizer_from_schedule')}")
                            #print(f"DEBUG: ✅ Successfully selected session for {day} {period}")
                        else:
                            #print(f"DEBUG: ⚠️ WARNING: AI selected session '{best_session_id}' not found in parallel_sessions!")
                            # Fallback to first session
                            best_session = parallel_sessions[0] if parallel_sessions else None
                            if best_session:
                                print(f"DEBUG: Using fallback session: {best_session['id']}")
                        print(f"{'='*60}\n")
                        
                        if best_session:
                            # Generate simplified LaTeX for this time slot with selected session
                            simplified_latex = self.generate_simplified_parallel_session_latex(time_slot, best_session)
                            personalized_schedule_parts.append({
                                'latex': simplified_latex,
                                'day_label': time_slot['day_label'],
                                'type': 'parallel_session'
                            })
                            selected_sessions_data.append(best_session)
                            
                            if IS_DEBUG_APP:
                                print(f"DEBUG: Selected {best_session.get('id')} for {time_slot['label']}")
                else:
                    # Keep non-parallel events as-is (plenary talks, breaks, etc.)
                    # These include check-in, opening ceremony, plenary sessions, breaks
                    personalized_schedule_parts.append({
                        'latex': time_slot['original_latex'],
                        'day_label': time_slot['day_label'],
                        'type': time_slot.get('type', 'unknown')
                    })
                    
                    if IS_DEBUG_APP:
                        print(f"DEBUG: Keeping non-parallel event: {time_slot.get('type', 'unknown')} - {time_slot['label']}")
            
            # Combine all parts into final schedule
            full_personalized_schedule = self.combine_schedule_parts_with_days(personalized_schedule_parts)
            
            # Store selected sessions for abstracts generation
            st.session_state.selected_parallel_sessions = selected_sessions_data
            
            return full_personalized_schedule
            
        except Exception as e:
            st.error(f"Error generating personalized schedule: {str(e)}")
            print(f"ERROR: {traceback.format_exc()}")
            return None

    def parse_schedule_time_slots(self, schedule_content: str) -> List[Dict]:
        """Parse Schedule.tex to identify time slots and parallel sessions in proper order"""
        time_slots = []
        
        # Find all day sections in the schedule
        # Look for patterns like "Mon, Jul 28, 2025 -- Morning" etc.
        day_pattern = r'\\TableHeading\{\s*\\hspace\*\{[^}]*\}([^}]+)\s*\}'
        day_matches = list(re.finditer(day_pattern, schedule_content))
        
        for i, day_match in enumerate(day_matches):
            day_label = day_match.group(1).strip()
            
            # Find the content for this day (until next day or end)
            start_pos = day_match.end()
            if i + 1 < len(day_matches):
                end_pos = day_matches[i + 1].start()
            else:
                end_pos = len(schedule_content)
            
            day_content = schedule_content[start_pos:end_pos]
            
            # Parse this day's content sequentially to preserve order
            daily_slots = self.parse_daily_content_sequentially(day_content, day_label)
            time_slots.extend(daily_slots)
        
        return time_slots

    def parse_daily_content_sequentially(self, day_content: str, day_label: str) -> List[Dict]:
        """Parse day content sequentially to preserve the original order from Schedule.tex"""
        slots = []
        
        # Split the content into lines and process sequentially
        lines = day_content.split('\n')
        current_block = []
        in_parallel_session_block = False
        parallel_session_info = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for TableEvent (conference events like check-in, breaks)
            if '\\TableEvent{' in line:
                slots.append({
                    'day_label': day_label,
                    'label': f"{day_label} - Event",
                    'has_parallel_sessions': False,
                    'original_latex': line,
                    'type': 'event'
                })
            
            # Check for OpeningClosingEvent (opening/closing ceremonies)
            elif '\\OpeningClosingEvent{' in line:
                slots.append({
                    'day_label': day_label,
                    'label': f"{day_label} - Ceremony",
                    'has_parallel_sessions': False,
                    'original_latex': line,
                    'type': 'ceremony'
                })
            
            # Check for plenary session input
            elif '\\input{sess' in line and 'P' in line:
                slots.append({
                    'day_label': day_label,
                    'label': f"{day_label} - Plenary Session",
                    'has_parallel_sessions': False,
                    'original_latex': line,
                    'type': 'plenary'
                })
            
            # Check for start of parallel session block
            elif '\\rowcolor{\\SessionTitleColor}' in line:
                in_parallel_session_block = True
                current_block = [line]
                
                # Continue reading until we get the full parallel session definition
                i += 1
                while i < len(lines) and not ('\\\\\\hline' in lines[i] and 'rowcolor' not in lines[i]):
                    current_block.append(lines[i])
                    i += 1
                
                # Now we have the complete parallel session block
                session_block_content = '\n'.join(current_block)
                parallel_session_info = self.extract_session_info_from_block(session_block_content)
                
                if parallel_session_info['sessions']:
                    slots.append({
                        'day_label': day_label,
                        'label': f"{day_label} - Parallel Sessions",
                        'has_parallel_sessions': True,
                        'sessions': parallel_session_info['sessions'],
                        'original_latex': session_block_content,
                        'type': 'parallel_sessions'
                    })
                
                in_parallel_session_block = False
                continue  # Skip the increment at the end since we already advanced i
            
            i += 1
        
        return slots

    def extract_session_info_from_block(self, session_block: str) -> Dict:
        """Extract session information from a parallel session block"""
        sessions = []
        
        # Look for \tableSpecialCL and \tableContributedCL commands
        special_session_pattern = r'\\tableSpecialCL\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}'
        contributed_session_pattern = r'\\tableContributedCL\{([^}]+)\}\s*\{([^}]+)\}\s*\{([^}]+)\}'
        
        # Extract special sessions
        for match in re.finditer(special_session_pattern, session_block):
            room, title, session_id, organizer = match.groups()
            sessions.append({
                'type': 'special',
                'id': session_id.strip(),
                'title': title.strip(),
                'room': room.strip(),
                'organizer': organizer.strip()
            })
        
        # Extract contributed/technical sessions
        for match in re.finditer(contributed_session_pattern, session_block):
            room, title, chair = match.groups()
            # For technical sessions, generate an ID from the title
            session_id = f"T{len([s for s in sessions if s['type'] == 'contributed']) + 1}"
            sessions.append({
                'type': 'contributed',
                'id': session_id,
                'title': title.strip(),
                'room': room.strip(),
                'chair': chair.strip()
            })
        
        return {
            'sessions': sessions,
            'time_description': 'Parallel Sessions'
        }

    def load_parallel_session_data(self, time_slot: Dict) -> List[Dict]:
        """Load detailed data for all parallel sessions in a time slot using hardcoded mappings"""
        parallel_sessions = []
        
        # Extract day and period from complex time_slot labels
        label = time_slot.get('label', '')
        print(f"DEBUG: load_parallel_session_data() processing label: '{label}'")
        
        # Parse complex labels to extract day and period
        day, period = self.extract_day_period_from_label(label)
        
        print(f"DEBUG: load_parallel_session_data() extracted day='{day}', period='{period}'")
        
        # Get session IDs from hardcoded mappings
        session_ids = self.discover_parallel_sessions_for_period(day, period)
        
        print(f"DEBUG: load_parallel_session_data() got {len(session_ids)} sessions: {session_ids}")
        
        for session_id in session_ids:
            print(f"DEBUG: Processing session {session_id}")
            
            # Determine session type (S = special, T = contributed)
            session_type = 'special' if session_id.startswith('S') else 'contributed'
            print(f"DEBUG: Session {session_id} classified as type: {session_type}")
            
            # Load session details
            session_data = self.load_session_details(session_id, session_type)
            print(f"DEBUG: load_session_details({session_id}, {session_type}) returned: {session_data is not None}")
            
            if session_data:
                print(f"DEBUG: Successfully loaded session data for {session_id}: '{session_data.get('title', 'No title')}'")
                
                # Load talks for this session
                session_talks = self.load_session_talks(session_id, session_type)
                print(f"DEBUG: Loaded {len(session_talks)} talks for session {session_id}")
                session_data['talks'] = session_talks
                
                parallel_sessions.append(session_data)
                print(f"DEBUG: Added session {session_id} to parallel_sessions list")
            else:
                print(f"WARNING: Could not load session data for {session_id}")
                print(f"DEBUG: Session {session_id} FILTERED OUT - session_data was None")
        
        print(f"DEBUG: load_parallel_session_data() returning {len(parallel_sessions)} sessions")
        return parallel_sessions
    
    def extract_day_period_from_label(self, label: str) -> tuple:
        """Extract day and period from various label formats"""
        if not label:
            return 'Unknown', 'Unknown'
        
        # Normalize the label
        label_lower = label.lower()
        
        # Extract day
        day = 'Unknown'
        day_mappings = {
            'monday': 'Monday',
            'mon': 'Monday',
            'tuesday': 'Tuesday', 
            'tue': 'Tuesday',
            'wednesday': 'Wednesday',
            'wed': 'Wednesday',
            'thursday': 'Thursday',
            'thu': 'Thursday',
            'friday': 'Friday',
            'fri': 'Friday'
        }
        
        for day_key, day_full in day_mappings.items():
            if day_key in label_lower:
                day = day_full
                break
        
        # Extract period
        period = 'Unknown'
        if 'morning' in label_lower:
            period = 'Morning'
        elif 'afternoon' in label_lower:
            period = 'Afternoon'
        elif 'parallel sessions' in label_lower or 'sessions' in label_lower:
            # For labels like "Monday - Parallel Sessions", we need to determine period
            # This is tricky without more context, so let's try both
            period = 'Morning'  # Default to morning, will be refined later
        
        print(f"DEBUG: extract_day_period_from_label('{label}') -> ('{day}', '{period}')")
        return day, period

    def load_session_details(self, session_id: str, session_type: str) -> Optional[Dict]:
        """Load detailed information for a specific session"""
        if session_type == 'special':
            # Load from special session abstract file (e.g., S1.tex, S2.tex)
            session_file = self.abstracts_path / f"{session_id}.tex"
            if session_file.exists():
                try:
                    content = self.safe_read_file(session_file)
                    if not content:
                        return None
                    return self.parse_session_abstract(content, session_id)
                except Exception as e:
                    print(f"Error loading session {session_id}: {e}")
                    
        elif session_type == 'contributed':
            # For technical sessions, create a basic structure
            return {
                'type': 'session',
                'id': session_id,
                'title': f"Technical Session {session_id}",
                'organizers': [],
                'description': "Technical session with contributed talks"
            }
        
        return None

    def load_session_talks(self, session_id: str, session_type: str) -> List[Dict]:
        """Load all talks for a specific session"""
        talks = []
        
        if session_type == 'special':
            # Load talks like S1-1.tex, S1-2.tex, etc.
            talk_counter = 1
            while True:
                talk_id = f"{session_id}-{talk_counter}"
                talk_file = self.abstracts_path / f"{talk_id}.tex"
                
                if not talk_file.exists():
                    break
                    
                try:
                    content = self.safe_read_file(talk_file)
                    if not content:
                        break
                    
                    talk_data = self.parse_talk_abstract(content, talk_id)
                    if talk_data:
                        talks.append(talk_data)
                        
                except Exception as e:
                    print(f"Error loading talk {talk_id}: {e}")
                
                talk_counter += 1
                
        elif session_type == 'contributed':
            # Load technical session talks (T1-1.tex, T1-2.tex, etc.)
            # Extract number from session_id (e.g., T1 -> 1)
            session_num = re.search(r'T(\d+)', session_id)
            if session_num:
                base_id = f"T{session_num.group(1)}"
                talk_counter = 1
                while True:
                    talk_id = f"{base_id}-{talk_counter}"
                    talk_file = self.abstracts_path / f"{talk_id}.tex"
                    
                    if not talk_file.exists():
                        break
                        
                    try:
                        content = self.safe_read_file(talk_file)
                        if not content:
                            break
                        
                        talk_data = self.parse_talk_abstract(content, talk_id)
                        if talk_data:
                            talks.append(talk_data)
                            
                    except Exception as e:
                        print(f"Error loading talk {talk_id}: {e}")
                    
                    talk_counter += 1
        
        return talks

    def generate_simplified_parallel_session_latex(self, time_slot: Dict, selected_session: Dict) -> str:
        """Generate simplified LaTeX for parallel sessions showing only the selected session"""
        session_id = selected_session['id']
        session_title = selected_session.get('title', selected_session.get('schedule_title', 'Unknown Session'))
        room = selected_session.get('room', 'TBD')
        
        # Clean and escape LaTeX special characters
        session_title = self.escape_latex(session_title)
        room = self.escape_latex(room)
        session_id = self.escape_latex(session_id)
        
        # Get organizer or chair
        organizer = ""
        if selected_session.get('organizers'):
            if isinstance(selected_session['organizers'], list) and selected_session['organizers']:
                organizer = selected_session['organizers'][0].get('name', '') if isinstance(selected_session['organizers'][0], dict) else str(selected_session['organizers'][0])
            else:
                organizer = str(selected_session['organizers'])
        elif selected_session.get('chair_from_schedule'):
            organizer = selected_session['chair_from_schedule']
        elif selected_session.get('organizer_from_schedule'):
            organizer = selected_session['organizer_from_schedule']
        
        # Clean organizer name
        organizer = self.escape_latex(organizer)
        
        # Create simplified LaTeX showing only the selected session
        # Format: Parallel Sessions [Selected Session Only]
        simplified_latex = f"""\\rowcolor{{\\SessionTitleColor}}
&\\multicolumn{{5}}{{|c|}}{{\\textbf{{Selected Session:}} {{{session_title}}} ({session_id}) - Room: {room}, Organizer: {organizer}}}
\\\\\\hline"""
        
        return simplified_latex

    def extract_original_time_slot_latex(self, schedule_content: str, time_slot: Dict) -> str:
        """Extract original LaTeX for non-parallel time slots"""
        return time_slot.get('original_latex', '')

    def combine_schedule_parts_with_days(self, schedule_parts: List[Dict]) -> str:
        """Combine all schedule parts into a complete personalized schedule with proper day headers"""
        
        # Start with the basic schedule structure
        full_schedule = """\\section{Personalized Conference Schedule}


"""
        
        # Track current day to add day headers when day changes
        current_day = None
        day_mapping = {
            'Mon': 'Monday, July 28, 2025',
            'Tue': 'Tuesday, July 29, 2025', 
            'Wed': 'Wednesday, July 30, 2025',
            'Thu': 'Thursday, July 31, 2025',
            'Fri': 'Friday, August 1, 2025'
        }
        
        current_list_open = False
        
        # Process each schedule part and convert to simple list format
        for i, part_info in enumerate(schedule_parts):
            if not part_info or not part_info.get('latex', '').strip():
                continue
                
            part = part_info['latex']
            day_label = part_info['day_label']
            part_type = part_info['type']
            
            # Extract the short day from day_label (e.g., "Mon, Jul 28, 2025 -- Morning" -> "Mon")
            day_short = day_label.split(',')[0].strip() if day_label else 'Unknown'
            
            # Check if we need to start a new day section
            if day_short != current_day:
                # Close previous day's list if open
                if current_list_open:
                    full_schedule += "\\end{itemize}\n\n"
                    current_list_open = False
                
                # Add new day header (consistent formatting for all days)
                full_day_name = day_mapping.get(day_short, day_label)
                full_schedule += f"\\subsection{{{full_day_name}}}\n\n"
                full_schedule += "\\begin{itemize}\n"
                current_list_open = True
                current_day = day_short
            
            # If list is not open (shouldn't happen with above logic, but safety check)
            if not current_list_open:
                full_schedule += "\\begin{itemize}\n"
                current_list_open = True
            
            # Convert complex table entries to simple list items
            if '\\TableEvent{' in part:
                # Extract event information
                event_match = re.search(r'\\TableEvent\{([^}]+)\}\{([^}]+)\}', part)
                if event_match:
                    time_info = event_match.group(1)
                    event_desc = event_match.group(2)
                    full_schedule += f"    \\item \\textbf{{{time_info}}} -- {event_desc}\n"
            
            elif '\\OpeningClosingEvent{' in part:
                # Extract ceremony information
                event_match = re.search(r'\\OpeningClosingEvent\{([^}]+)\}\{([^}]+)\}', part)
                if event_match:
                    time_info = event_match.group(1)
                    event_desc = event_match.group(2)
                    full_schedule += f"    \\item \\textbf{{{time_info}}} -- {event_desc}\n"
            
            elif '\\input{sess' in part:
                # Handle plenary session input
                if 'P' in part:
                    # Extract the plenary session file (e.g., sessP1.tex)
                    input_match = re.search(r'\\input\{([^}]+)\}', part)
                    if input_match:
                        sess_file = input_match.group(1)
                        # Try to read and parse the plenary session file to extract speaker and title
                        try:
                            sess_path = os.path.join(self.tex_dir, sess_file)
                            if os.path.exists(sess_path):
                                sess_content = self.safe_read_file(sess_path)
                                if sess_content:
                                    # Parse \\tablePlenary content to extract speaker and title info
                                    plenary_match = re.search(r'\\tablePlenary\{([^}]+)\}.*?\{([^}]+)\}.*?\{([^}]+)\}.*?\{([^}]+)\}.*?\{([^}]+)\}.*?\{([^}]+)\}', sess_content, re.DOTALL)
                                if plenary_match:
                                    time_slot = plenary_match.group(1)
                                    room = plenary_match.group(2) 
                                    chair = plenary_match.group(3)
                                    speaker_info = plenary_match.group(4)
                                    talk_title = plenary_match.group(5)
                                    talk_id = plenary_match.group(6)
                                    
                                    # Extract speaker name from speaker_info (format: "Name, Affiliation, Title")
                                    speaker_parts = speaker_info.split(',')
                                    speaker_name = speaker_parts[0].strip() if speaker_parts else "Unknown Speaker"
                                    
                                    full_schedule += f"    \\item \\textbf{{{time_slot}}} -- \\textbf{{Plenary Talk}}: {talk_title}, {speaker_name}, {room}, {chair}\n"
                                else:
                                    full_schedule += f"    \\item \\textbf{{Plenary Session}} -- Details in {sess_file}\n"
                            else:
                                full_schedule += f"    \\item \\textbf{{Plenary Session}} -- See full program book for details\n"
                        except Exception as e:
                            full_schedule += f"    \\item \\textbf{{Plenary Session}} -- See full program book for details\n"
                    else:
                        full_schedule += f"    \\item \\textbf{{Plenary Session}} -- See full program book for details\n"
            
            elif 'Selected Session:' in part:
                # Handle our AI-selected parallel sessions
                session_match = re.search(r'\\textbf\{Selected Session:\}\s*\{([^}]+)\}\s*\(([^)]+)\)\s*-\s*Room:\s*([^,]+),\s*Organizer:\s*([^}]+)', part)
                if session_match:
                    session_title = session_match.group(1)
                    session_id = session_match.group(2)
                    room = session_match.group(3)
                    organizer = session_match.group(4).rstrip('}')
                    full_schedule += f"    \\item \\textbf{{Parallel Sessions}} -- AI Selected: {session_title} ({session_id}), {room}, {organizer}\n"
                else:
                    # Fallback for any other parallel session format
                    full_schedule += f"    \\item \\textbf{{Parallel Sessions}} -- AI selected based on your preferences\n"
            
            else:
                # For any other content, try to extract meaningful information
                if part.strip():
                    # Remove LaTeX table formatting and extract text
                    clean_part = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', part)
                    clean_part = re.sub(r'\\[a-zA-Z]+\*?', '', clean_part)
                    clean_part = re.sub(r'[{}\\&|]', '', clean_part)
                    clean_part = clean_part.strip()
                    if clean_part and len(clean_part) > 5:
                        full_schedule += f"    \\item {clean_part}\n"
        
        # Close the final list
        if current_list_open:
            full_schedule += "\\end{itemize}\n\n"
        
        full_schedule += "\\vspace{1cm}\n\n"
        
        return full_schedule

    def combine_schedule_parts(self, schedule_parts: List[str]) -> str:
        """Combine all schedule parts into a complete personalized schedule"""
        
        # Start with the basic schedule structure that doesn't require complex table formatting
        full_schedule = """\\section{Personalized Conference Schedule}


\\subsection{Monday, July 28, 2025}

\\begin{itemize}
"""
        
        # Process each schedule part and convert to simple list format
        for i, part in enumerate(schedule_parts):
            if part.strip():  # Only add non-empty parts
                # Convert complex table entries to simple list items
                if '\\TableEvent{' in part:
                    # Extract event information
                    event_match = re.search(r'\\TableEvent\{([^}]+)\}\{([^}]+)\}', part)
                    if event_match:
                        time_info = event_match.group(1)
                        event_desc = event_match.group(2)
                        full_schedule += f"    \\item \\textbf{{{time_info}}} -- {event_desc}\n"
                
                elif '\\OpeningClosingEvent{' in part:
                    # Extract ceremony information
                    event_match = re.search(r'\\OpeningClosingEvent\{([^}]+)\}\{([^}]+)\}', part)
                    if event_match:
                        time_info = event_match.group(1)
                        event_desc = event_match.group(2)
                        full_schedule += f"    \\item \\textbf{{{time_info}}} -- {event_desc}\n"
                
                elif '\\input{sess' in part:
                    # Handle plenary session input
                    if 'P' in part:
                        # Extract the plenary session file (e.g., sessP1.tex)
                        input_match = re.search(r'\\input\{([^}]+)\}', part)
                        if input_match:
                            sess_file = input_match.group(1)
                            # Try to read and parse the plenary session file to extract speaker and title
                            try:
                                sess_path = os.path.join(self.tex_dir, sess_file)
                                if os.path.exists(sess_path):
                                    sess_content = self.safe_read_file(sess_path)
                                    if sess_content:
                                        # Parse \tablePlenary content to extract speaker and title info
                                        plenary_match = re.search(r'\\tablePlenary\{([^}]+)\}.*?\{([^}]+)\}.*?\{([^}]+)\}.*?\{([^}]+)\}.*?\{([^}]+)\}.*?\{([^}]+)\}', sess_content, re.DOTALL)
                                    if plenary_match:
                                        time_slot = plenary_match.group(1)
                                        room = plenary_match.group(2) 
                                        chair = plenary_match.group(3)
                                        speaker_info = plenary_match.group(4)
                                        talk_title = plenary_match.group(5)
                                        talk_id = plenary_match.group(6)
                                        
                                        # Extract speaker name from speaker_info (format: "Name, Affiliation, Title")
                                        speaker_parts = speaker_info.split(',')
                                        speaker_name = speaker_parts[0].strip() if speaker_parts else "Unknown Speaker"
                                        
                                        full_schedule += f"    \\item \\textbf{{{time_slot}}} -- \\textbf{{Plenary Talk}}: {talk_title}, {speaker_name}, {room},  {chair}\n"
                                    else:
                                        full_schedule += f"    \\item \\textbf{{Plenary Session}} -- Details in {sess_file}\n"
                                else:
                                    full_schedule += f"    \\item \\textbf{{Plenary Session}} -- See full program book for details\n"
                            except Exception as e:
                                full_schedule += f"    \\item \\textbf{{Plenary Session}} -- See full program book for details\n"
                        else:
                            full_schedule += f"    \\item \\textbf{{Plenary Session}} -- See full program book for details\n"
                
                elif 'Selected Session:' in part:
                    # Handle our AI-selected parallel sessions
                    session_match = re.search(r'\\textbf\{Selected Session:\}\s*\{([^}]+)\}\s*\(([^)]+)\)\s*-\s*Room:\s*([^,]+),\s*Organizer:\s*([^}]+)', part)
                    if session_match:
                        session_title = session_match.group(1)
                        session_id = session_match.group(2)
                        room = session_match.group(3)
                        organizer = session_match.group(4).rstrip('}')
                        full_schedule += f"    \\item \\textbf{{Parallel Sessions}} -- AI Selected: {session_title} ({session_id})\n"
                        full_schedule += f"          \\\\Room: {room}, Organizer: {organizer}\n"
                    else:
                        # Fallback for any other parallel session format
                        full_schedule += f"    \\item \\textbf{{Parallel Sessions}} -- AI selected based on your preferences\n"
                
                else:
                    # For any other content, try to extract meaningful information
                    if part.strip():
                        # Remove LaTeX table formatting and extract text
                        clean_part = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', part)
                        clean_part = re.sub(r'\\[a-zA-Z]+\*?', '', clean_part)
                        clean_part = re.sub(r'[{}\\&|]', '', clean_part)
                        clean_part = clean_part.strip()
                        if clean_part and len(clean_part) > 5:
                            full_schedule += f"    \\item {clean_part}\n"
        
        # Close the list and add note
        full_schedule += """\\end{itemize}

\\vspace{1cm}



"""
        
        return full_schedule

    def generate_personalized_abstracts(self, selected_sessions: List[Dict]) -> str:
        """Generate abstracts section from selected sessions using raw LaTeX input files"""
        abstracts_content = []
        
        # Add header for abstracts section
        abstracts_content.append("\\section{Selected Talk Abstracts}")
        abstracts_content.append("")
        abstracts_content.append("This section contains information for sessions in your personalized schedule.")
        abstracts_content.append("")
        
        # First, create all session labels upfront to resolve references
        abstracts_content.append("% Creating all session labels upfront to resolve references")
        for session in selected_sessions:
            session_id = session.get('id', 'Unknown')
            abstracts_content.append(f"\\sessionlabel{{{session_id}}}")
        abstracts_content.append("")
        
        # Add abstracts for each selected session using \input commands
        for i, session in enumerate(selected_sessions):
            session_id = session.get('id', 'Unknown')
            session_title = session.get('title', f'Session {i+1}')
            
            # Create section for the session
            abstracts_content.append(f"\\subsection{{Session {i+1} – {session_id}: {self.escape_latex(session_title)}}}")
            abstracts_content.append("")
            
            # Input the session schedule file from MCM_ProgramBook_TEX
            abstracts_content.append(f"% Including consolidated session talks file via \\input")
            abstracts_content.append(f"% Note: Using modified environment to handle missing parameters")
            abstracts_content.append("% Redefine talk environment to handle 6-parameter format from consolidated files")
            abstracts_content.append("\\let\\originaltalk\\talk")
            abstracts_content.append("\\let\\originalendtalk\\endtalk")
            abstracts_content.append("\\renewenvironment{talk}[6]{%")
            abstracts_content.append(f"  \\originaltalk{{#1}}{{#2}}{{#3}}{{#4}}{{#5}}{{#6}}{{TBD}}{{talk-{session_id}-\\thetalk}}{{{session_id}}}")
            abstracts_content.append("}{\\originalendtalk}")
            abstracts_content.append(f"\\input{{{session_id}_sess_talks}}")
            abstracts_content.append("% Restore original talk environment")
            abstracts_content.append("\\renewenvironment{talk}[9]{\\originaltalk{#1}{#2}{#3}{#4}{#5}{#6}{#7}{#8}{#9}}{\\originalendtalk}")
            abstracts_content.append("")
            
            # Input the session abstract file from MCM_ProgramBook_TEX
            mcm_session_path = f"{self.base_path.parent}/preprocess/abstracts/{session_id}.tex"
            if os.path.exists(mcm_session_path):
                abstracts_content.append(f"% Including session abstract file via \\input")
                abstracts_content.append(f"\\input{{{session_id}}}")
                abstracts_content.append("")
            
            # Input each talk file for this session (typically 1-4 talks)
            for talk_num in range(1, 5):
                talk_id_full = f"{session_id}-{talk_num}"
                # Check if the file exists in MCM_ProgramBook_TEX directory
                mcm_tex_path = f"{self.base_path.parent}/preprocess/abstracts/{talk_id_full}.tex"
                if os.path.exists(mcm_tex_path):
                    abstracts_content.append(f"\\input{{{talk_id_full}}}")
            
            abstracts_content.append("")
            abstracts_content.append("\\newpage")
            abstracts_content.append("")
        
        return "\n".join(abstracts_content)

    def get_plenary_abstracts(self) -> List[str]:
        """Get abstracts for all plenary talks"""
        plenary_abstracts = []
        
        # Look for plenary talk files (P1.tex, P2.tex, etc.)
        plenary_pattern = re.compile(r'^P\d+\.tex$')
        
        if self.abstracts_path.exists():
            for file in self.abstracts_path.iterdir():
                if plenary_pattern.match(file.name):
                    try:
                        content = self.safe_read_file(file)
                        if content:
                            talk_data = self.parse_talk_abstract(content, file.stem)
                            if talk_data:
                                talk_latex = self.format_talk_abstract_for_latex(talk_data)
                                plenary_abstracts.append(talk_latex)
                            
                    except Exception as e:
                        print(f"Error loading plenary abstract {file.name}: {e}")
        
        return plenary_abstracts

    def format_talk_abstract_for_latex(self, talk_data: Dict) -> str:
        """Format a talk abstract for inclusion in LaTeX document"""
        content = []
        
        # Title
        title = self.escape_latex(talk_data.get('title', 'Unknown Title'))
        content.append(f"\\textbf{{{title}}}")
        content.append("")
        
        # Speaker and affiliation
        speaker = self.escape_latex(talk_data.get('speaker', 'Unknown Speaker'))
        affiliation = self.escape_latex(talk_data.get('affiliation', ''))
        
        content.append(f"\\textit{{{speaker}}}")
        if affiliation:
            content.append(f"\\\\{affiliation}")
        content.append("")
        
        # Email
        email = talk_data.get('email', '')
        if email:
            content.append(f"\\texttt{{{email}}}")
            content.append("")
        
        # Coauthors
        coauthors = talk_data.get('coauthors', '')
        if coauthors:
            coauthors_clean = self.escape_latex(coauthors)
            content.append(f"\\textbf{{Coauthors:}} {coauthors_clean}")
            content.append("")
        
        # Abstract
        abstract = talk_data.get('abstract', '')
        if abstract:
            abstract_clean = self.escape_latex(abstract)
            content.append(abstract_clean)
        else:
            content.append("\\textit{No abstract available.}")
        
        content.append("\\vspace{0.5cm}")
        
        return "\n".join(content)

    def create_personalized_latex_document(self, schedule_latex: str, abstracts_latex: str, user_interests: Dict) -> str:
        """Create complete LaTeX document with personalized schedule and abstracts"""
        
        # Use LaTeX header that matches MCM book format with book class
        header = """\\documentclass[12pt,a4paper,oneside]{book}

\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{amsmath,amssymb}
\\usepackage{geometry}
\\usepackage{xcolor}
\\usepackage{enumitem}
\\usepackage{fancyhdr}
\\usepackage{needspace}
\\usepackage{url}
\\usepackage[colorlinks=false]{hyperref}
\\usepackage{mcm_macros}

\\geometry{margin=1in}

% URL formatting to prevent overfull hbox
\\def\\UrlBreaks{\\do\\/\\do\\-\\do\\.\\do\\=\\do\\&}

% Simple macro to handle session references safely - just creates label and phantom text
\\newcommand{\\sessionlabel}[1]{%
  \\label{#1}%
  \\phantom{Session #1}%
}

% Redefine pageref to be more forgiving for undefined references
\\let\\originalpageref\\pageref
\\renewcommand{\\pageref}[1]{%
  \\@ifundefined{r@#1}{\\textbf{??}}{\\originalpageref{#1}}%
}

\\title{MCM 2025 Personalized Conference Schedule}
\\author{Generated by AI Assistant}
\\date{\\today}

\\begin{document}

\\maketitle
"""
        
        # Create title page content
        title_content = self.create_personalized_title_page(user_interests)
        
        # Combine all content with simpler structure
        full_document = f"""{header}

{title_content}

{schedule_latex}

{abstracts_latex}

\\end{{document}}"""
        
        return full_document

    def create_personalized_title_page(self, user_interests: Dict) -> str:
        """Create a personalized title page"""
        keywords = ', '.join(user_interests.get('keywords', []))
        preferences = user_interests.get('preferences', '')
        
        title_page = f"""
\\section{{Your Input}}

\\textbf{{Research Keywords:}} {self.escape_latex(keywords)}

"""

        # Add preferences section if they exist
        if preferences.strip():
            title_page += f"""\\textbf{{Additional Preferences:}}

{self.escape_latex(preferences)}

"""
        else:
            title_page += """\\textbf{{Additional Preferences:}} None specified

"""

        return title_page

    def compile_and_download_personalized_pdf(self, latex_content: str, user_interests: Dict):
        """Compile LaTeX to PDF and provide download, saving both source and PDF to MCM-2025-Program-Personal directory"""
        try:
            # Generate simple, safe filename components
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_name = getattr(self, 'ai_model_name', st.session_state.get('selected_model', 'unknown'))
            # Simplified filename without complex keywords
            base_filename = f"MCM2025_AIProgram_{timestamp}_{model_name}"
            
            # Use personal directory inside the main project directory
            personal_output_path = self.base_path.parent / "MCM-2025-Program-Personal"
            personal_output_path.mkdir(exist_ok=True)
            
            # Save LaTeX source to personal directory
            tex_filename = f"{base_filename}.tex"
            tex_output_path = personal_output_path / tex_filename
            
            with open(tex_output_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            print(f"💾 LaTeX source saved to: {tex_output_path}")
            
            # Create a temporary directory for compilation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write LaTeX content to temporary file for compilation
                temp_tex_file = temp_path / "personalized_schedule.tex"
                with open(temp_tex_file, 'w', encoding='utf-8') as f:
                    f.write(latex_content)
                
                # Copy necessary style files to temp directory
                mcm_macros_path = self.output_path / "mcm_macros.sty"
                if mcm_macros_path.exists():
                    shutil.copy(mcm_macros_path, temp_path / "mcm_macros.sty")
                
                # Compile with pdflatex
                compile_command = f"cd {temp_path} && pdflatex -interaction=nonstopmode personalized_schedule.tex"
                
                with st.spinner("📄 Compiling personalized schedule to PDF..."):
                    try:
                        result = subprocess.run(
                            compile_command,
                            shell=True,
                            capture_output=True,
                            text=False,  # Get bytes instead of text to handle encoding issues
                            timeout=120
                        )
                        
                        # Decode output safely, replacing problematic characters
                        try:
                            stdout = result.stdout.decode('utf-8', errors='replace')
                            stderr = result.stderr.decode('utf-8', errors='replace')
                        except:
                            stdout = str(result.stdout)
                            stderr = str(result.stderr)
                            
                    except subprocess.TimeoutExpired:
                        st.error("❌ PDF compilation timed out after 2 minutes")
                        return
                    except Exception as e:
                        st.error(f"❌ Error running pdflatex: {str(e)}")
                        return
                
                temp_pdf_file = temp_path / "personalized_schedule.pdf"
                
                if temp_pdf_file.exists():
                    # Copy PDF to personal directory
                    pdf_filename = f"{base_filename}.pdf"
                    pdf_output_path = personal_output_path / pdf_filename
                    shutil.copy(temp_pdf_file, pdf_output_path)
                    
                    st.success(f"✅ PDF saved to: {pdf_output_path}")
                    
                    # Read PDF content for download
                    with open(temp_pdf_file, 'rb') as f:
                        pdf_content = f.read()
                    
                    # Provide download button
                    st.download_button(
                        label="📥 Download Personalized Schedule PDF",
                        data=pdf_content,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    # Show summary
                    selected_sessions = st.session_state.get('selected_parallel_sessions', [])
                    if selected_sessions:
                        st.info(f"📊 **Summary:** Selected {len(selected_sessions)} sessions based on your interests")
                        
                        with st.expander("View Selected Sessions"):
                            for session in selected_sessions:
                                session_id = session.get('id', 'Unknown')
                                session_title = session.get('title', 'Unknown Session')
                                reasoning = session.get('ai_selection_reasoning', 'No reasoning available')
                                
                                st.write(f"**{session_id}**: {session_title}")
                                st.write(f"*AI Reasoning*: {reasoning}")
                                st.write("---")
                
                else:
                    st.error("❌ PDF compilation failed")
                    st.info(f"💾 LaTeX source is still available at: {tex_output_path}")
                    
                    # Show compilation errors in detail using safely decoded output
                    if 'stderr' in locals() and stderr:
                        with st.expander("LaTeX Compilation Error Details"):
                            st.code(stderr)
                    
                    if 'stdout' in locals() and stdout:
                        with st.expander("LaTeX Compilation Output"):
                            st.code(stdout)
                    
                    # Show the LaTeX log file if it exists
                    log_file = temp_path / "personalized_schedule.log"
                    if log_file.exists():
                        try:
                            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                                log_content = f.read()
                            with st.expander("LaTeX Log File"):
                                st.code(log_content)
                        except Exception:
                            st.info("LaTeX log file exists but couldn't be read safely")
                    
                    # Offer LaTeX download as fallback (even though it's already saved)
                    st.download_button(
                        label="📥 Download LaTeX Source (for manual compilation)",
                        data=latex_content,
                        file_name=tex_filename,
                        mime="text/plain",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"Error during PDF compilation: {str(e)}")
            print(f"PDF compilation error: {traceback.format_exc()}")

    def get_latex_header(self) -> str:
        """Get standard LaTeX document header with necessary packages and macros"""
        return """\\documentclass[12pt,a4paper,figuresright]{book}

\\usepackage{amsmath,amssymb}
\\usepackage{tabularx,multirow,graphicx,url,wrapfig,xcolor,rotating,multicol,epsfig,colortbl}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}

\\setlength{\\textheight}{25.2cm}
\\setlength{\\textwidth}{16.5cm}
\\setlength{\\voffset}{-1.6cm}
\\setlength{\\hoffset}{-0.3cm}
\\setlength{\\evensidemargin}{-0.3cm}
\\setlength{\\oddsidemargin}{0.3cm}
\\setlength{\\parindent}{0cm}
\\setlength{\\parskip}{0.3cm}

% Table formatting macros
\\newcommand{\\numcols}{5}
\\definecolor{SessionTitleColor}{RGB}{128,128,128}
\\definecolor{SessionLightColor}{RGB}{220,220,220}

% Define Y column type for tabularx
\\newcolumntype{Y}{>{\\centering\\arraybackslash}X}

\\newcommand{\\TableHeading}[1]{\\multicolumn{6}{|c|}{\\cellcolor{SessionTitleColor}\\textcolor{white}{\\textbf{#1}}}}
\\newcommand{\\TableEvent}[2]{\\multicolumn{6}{|c|}{\\cellcolor{SessionLightColor}\\textbf{#1} -- #2}}
\\newcommand{\\OpeningClosingEvent}[2]{\\multicolumn{6}{|c|}{\\cellcolor{SessionTitleColor}\\textcolor{white}{\\textbf{#1} -- #2}}}

\\newcommand{\\tableSpecialCL}[4]{\\begin{minipage}[t]{\\linewidth}\\centering\\textbf{#1}\\\\#2\\\\Session #3\\\\Organizer: #4\\end{minipage}}
\\newcommand{\\tableContributedCL}[3]{\\begin{minipage}[t]{\\linewidth}\\centering\\textbf{#1}\\\\#2\\\\Chair: #3\\end{minipage}}
\\newcommand{\\tableTalk}[3]{\\begin{minipage}[t]{\\linewidth}#2\\\\\\textit{#1}\\\\Talk ID: #3\\end{minipage}}
\\newcommand{\\tableTime}[2]{\\textbf{#1--#2}}

\\title{MCM 2025 Personalized Schedule}
\\date{\\today}

\\begin{document}"""


def main():
    """Main function to run the Streamlit app"""
    app = MCMStreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
