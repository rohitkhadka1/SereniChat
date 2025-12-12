from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_core.runnables import RunnableSequence
from langchain.llms.base import LLM
import os
import logging
import json
import time
from typing import Generator, Optional, Dict, Any, List, Tuple
from .error_handler import APIError, ValidationError, ConfigurationError
from dotenv import load_dotenv

load_dotenv()

# Configure logging with more appropriate level
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SimpleOutputParser(BaseOutputParser):
    """Simple output parser that returns the text as-is."""
    
    def parse(self, text: str) -> str:
        """Parse the output text."""
        return text.strip() if text else ""
    
    @property
    def _type(self) -> str:
        """Return parser type."""
        return "simple"


class FallbackLLM(LLM):
    """Fallback LLM for when API calls fail."""
    
    @property
    def _llm_type(self) -> str:
        """Return LLM type."""
        return "fallback"
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        **kwargs: Any
    ) -> str:
        """Generate response based on keywords."""
        prompt_lower = prompt.lower()
        
        responses = {
            'anxiety': "I understand you're experiencing anxiety. Try deep breathing exercises: breathe in for 4 counts, hold for 4, exhale for 4. Consider speaking with a mental health professional for personalized support.",
            'depression': "I hear that you're going through a difficult time. Depression is treatable, and you don't have to face it alone. Please consider reaching out to a mental health professional or counselor who can provide proper support.",
            'stress': "Stress can be overwhelming. Try breaking tasks into smaller steps, practice mindfulness, and ensure you're getting adequate rest. If stress persists, consider speaking with a counselor.",
            'sleep': "Good sleep is crucial for mental health. Try maintaining a regular sleep schedule, avoiding screens before bed, and creating a calm sleep environment. If sleep issues persist, consult a healthcare provider.",
        }
        
        for keyword, response in responses.items():
            if keyword in prompt_lower:
                return response
        
        return "Thank you for sharing. Mental health is important, and it's okay to seek help. Consider speaking with a mental health professional who can provide personalized guidance for your situation."


class EnhancedFallbackLLM(LLM):
    """Enhanced fallback LLM that provides more contextual responses."""
    
    @property
    def _llm_type(self) -> str:
        """Return LLM type."""
        return "enhanced_fallback"
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        **kwargs: Any
    ) -> str:
        """Generate enhanced contextual response."""
        prompt_lower = prompt.lower()
        
        # Extract context and input from prompt
        context = ""
        user_input = ""
        
        if "Context:" in prompt and "Question:" in prompt:
            parts = prompt.split("Question:")
            if len(parts) >= 2:
                context = parts[0].replace("Context:", "").strip()
                user_input = parts[1].replace("Answer:", "").strip()
        else:
            user_input = prompt
        
        # Generate contextual response
        return self._generate_contextual_response(user_input, context)
    
    def _generate_contextual_response(self, user_input: str, context: str) -> str:
        """Generate a contextual response based on user input and available context."""
        user_lower = user_input.lower()
        
        # Enhanced response patterns
        if any(word in user_lower for word in ['anxiety', 'anxious', 'panic', 'worried']):
            if 'interview' in user_lower or 'job' in user_lower:
                return """I understand you're feeling anxious about your job interview. This is completely normal! Here are some specific strategies for interview anxiety:

• Practice your responses to common questions out loud
• Prepare questions to ask the interviewer - it shows engagement
• Visualize a successful interview the night before
• Arrive 10-15 minutes early to settle in
• Remember: they want to find the right person, not to trick you
• Take deep breaths before entering the room

You've got this! Your preparation and authenticity are your strengths."""
            
            elif 'attack' in user_lower:
                return """When experiencing anxiety attacks, try these immediate techniques:

1. **Grounding (5-4-3-2-1 technique)**: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste
2. **Breathing**: Inhale for 4 counts, hold for 4, exhale for 6 counts
3. **Remind yourself**: "This is temporary, I am safe, this will pass"
4. **Find a quiet space** if possible
5. **Use cold water** on your wrists or face

If attacks are frequent, please consult a mental health professional for personalized coping strategies."""
            
            else:
                return """I understand you're dealing with anxiety. Here are some evidence-based strategies that can help:

**Immediate relief:**
• Deep breathing exercises (4-4-6 pattern)
• Progressive muscle relaxation
• Mindfulness meditation (even 5 minutes helps)

**Long-term management:**
• Regular exercise (even walking helps)
• Consistent sleep schedule
• Limit caffeine and alcohol
• Practice grounding techniques
• Consider therapy or counseling

Remember, anxiety is treatable and you don't have to face it alone. Professional support can provide personalized strategies."""
        
        elif any(word in user_lower for word in ['depression', 'depressed', 'sad', 'hopeless']):
            return """I hear that you're going through a difficult time. Depression affects many people and is treatable. Here's what can help:

**Immediate support:**
• Reach out to trusted friends or family
• Maintain daily routines (even small ones)
• Get sunlight and fresh air when possible
• Practice self-compassion - be kind to yourself

**Professional help:**
• Therapy and counseling are highly effective
• Medication can be helpful when combined with therapy
• Support groups provide community and understanding

**Important:** If you're having thoughts of self-harm, please contact a crisis helpline immediately. You matter, and there are people who want to help you through this.

The National Suicide Prevention Lifeline is 988 (US) or your local crisis number."""
        
        elif any(word in user_lower for word in ['stress', 'stressed', 'overwhelmed']):
            return """Feeling stressed or overwhelmed is very common. Here are some practical strategies:

**Immediate stress relief:**
• Take 5 deep breaths
• Step away for a 10-minute break
• Listen to calming music
• Do a quick body scan meditation

**Stress management:**
• Break large tasks into smaller steps
• Use time-blocking for your schedule
• Practice saying "no" to non-essential commitments
• Regular physical activity (even 10 minutes helps)
• Maintain boundaries between work and personal time

**Long-term strategies:**
• Identify your stress triggers
• Develop healthy coping mechanisms
• Consider stress management therapy
• Build a support network

Remember, some stress is normal, but if it's significantly impacting your life, professional help can provide personalized strategies."""
        
        elif any(word in user_lower for word in ['sleep', 'insomnia', 'tired']):
            return """Sleep issues can significantly impact mental health. Here are some evidence-based sleep strategies:

**Sleep hygiene:**
• Go to bed and wake up at the same time daily
• Create a cool, dark, quiet bedroom
• Avoid screens 1 hour before bed
• No caffeine after 2 PM
• Regular exercise (but not close to bedtime)

**Bedtime routine:**
• Wind down with calming activities
• Try relaxation techniques (progressive muscle relaxation)
• Keep a worry journal to clear your mind
• Use the bed only for sleep and intimacy

**If sleep problems persist:**
• Consult a healthcare provider
• Consider sleep therapy (CBT-I)
• Rule out underlying conditions
• Track your sleep patterns

Good sleep is foundational for mental health - it's worth investing in these habits."""
        
        # Use context if available
        elif context and "mental health" in context.lower():
            return f"""Based on the mental health information available, I want to provide you with supportive guidance. 

While I can offer general mental health information, everyone's situation is unique. I encourage you to speak with a qualified mental health professional who can provide personalized support and treatment options tailored to your specific needs.

**What you shared:** {user_input}

**Key points to remember:**
• Your feelings are valid and important
• Seeking help is a sign of strength, not weakness
• Mental health professionals are trained to help
• There are effective treatments available
• You don't have to face this alone

Please know that there are people and resources available to support you."""
        
        else:
            return f"""Thank you for sharing: "{user_input}"

I'm here to provide mental health support and information. While I'd like to give you more specific guidance, I want to ensure you receive the most appropriate help for your situation.

**I encourage you to:**
• Reach out to a qualified mental health professional
• Talk to your primary care doctor
• Contact a mental health helpline
• Connect with trusted friends or family

**If you're in crisis or having thoughts of self-harm:**
Please contact a crisis helpline or emergency services immediately. You don't have to face this alone.

**Remember:** Seeking help is a sign of strength, and there are people who want to support you through whatever you're experiencing."""


def get_conversation_chain() -> RunnableSequence:
    """
    Create a simple runnable chain with custom prompt template using Hugging Face models.
    
    Returns:
        RunnableSequence: The configured conversation chain
        
    Raises:
        ConfigurationError: If chain initialization fails
    """
    try:
        # Get and validate API key
        api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if not api_key:
            raise ConfigurationError(
                "HUGGINGFACEHUB_API_TOKEN environment variable is not set",
                details={"required_env_var": "HUGGINGFACEHUB_API_TOKEN"}
            )
        
        # Initialize LLM with fallback options
        llm = _initialize_llm(api_key)
        
        # Create prompt template
        template = """You are a helpful mental health support assistant. Answer the user's question using the provided context.

Context: {context}

Question: {input}

Answer: Provide a helpful, empathetic response that addresses the user's specific concern. Keep it concise and supportive."""
        
        prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=template,
            validate_template=True
        )
        
        # Create output parser
        output_parser = SimpleOutputParser()
        
        # Create the runnable chain with error handling
        try:
            chain = prompt | llm | output_parser
        except Exception as e:
            logger.warning(f"Failed to create chain with pipe operator: {e}")
            # Try alternative chain creation
            from langchain.chains import LLMChain
            chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
        
        if chain is None:
            raise ConfigurationError("Failed to create runnable chain")
        
        # Wrap chain with StopIteration protection
        class SafeChainWrapper:
            def __init__(self, chain):
                self.chain = chain
                self.llm = llm
                self.prompt = prompt
                self.output_parser = output_parser
            
            def invoke(self, inputs):
                try:
                    # Try normal chain invocation
                    return self.chain.invoke(inputs)
                except StopIteration:
                    # Handle StopIteration by invoking components separately
                    logger.debug("Handling StopIteration in chain invocation")
                    try:
                        formatted_prompt = self.prompt.format(**inputs)
                        llm_output = self.llm.invoke(formatted_prompt) if hasattr(self.llm, 'invoke') else self.llm(formatted_prompt)
                        return self.output_parser.parse(llm_output) if llm_output else None
                    except:
                        return None
                except Exception as e:
                    logger.debug(f"Chain invoke error: {e}")
                    return None
            
            def batch(self, inputs_list):
                try:
                    return self.chain.batch(inputs_list)
                except:
                    return [self.invoke(inputs) for inputs in inputs_list]
            
            def __getattr__(self, name):
                return getattr(self.chain, name)
        
        wrapped_chain = SafeChainWrapper(chain)
        logger.info("Runnable chain created successfully with safety wrapper")
        return wrapped_chain
        
    except Exception as e:
        logger.error(f"Error creating conversation chain: {str(e)}")
        raise ConfigurationError(
            "Failed to initialize conversation chain",
            details={"error": str(e)}
        )


def _initialize_llm(api_key: str) -> Any:
    """
    Initialize LLM with fallback options using direct API calls.
    
    Args:
        api_key: Hugging Face API token
        
    Returns:
        Initialized LLM instance
    """
    # Try using direct HuggingFace API calls to bypass StopIteration issues
    try:
        from huggingface_hub import InferenceClient
        
        class DirectHuggingFaceLLM:
            def __init__(self, api_key: str, model_name: str = "microsoft/DialoGPT-medium"):
                self.client = InferenceClient(token=api_key)
                self.model_name = model_name
                
            def invoke(self, prompt: str, **kwargs) -> str:
                try:
                    # Use the inference client directly
                    response = self.client.text_generation(
                        prompt=prompt,
                        model=self.model_name,
                        max_new_tokens=kwargs.get('max_new_tokens', 150),
                        temperature=kwargs.get('temperature', 0.7),
                        do_sample=True,
                        return_full_text=False
                    )
                    return response if response else ""
                except Exception as e:
                    logger.debug(f"Direct API call failed: {e}")
                    return ""
            
            def __call__(self, prompt: str, **kwargs) -> str:
                return self.invoke(prompt, **kwargs)
        
        # Test the direct API approach
        direct_llm = DirectHuggingFaceLLM(api_key)
        test_result = direct_llm.invoke("Hello")
        
        if test_result and test_result.strip():
            logger.info("Successfully initialized direct HuggingFace API client")
            return direct_llm
        else:
            logger.warning("Direct API test returned empty result")
            
    except Exception as e:
        logger.warning(f"Direct API approach failed: {str(e)}")
    
    # Try alternative models with different configurations
    models_to_try = [
        {
            "repo_id": "google/gemma-2-2b-it",
            "max_new_tokens": 100,
            "temperature": 0.7,
            "timeout": 30
        },
        {
            "repo_id": "distilbert-base-uncased",
            "max_new_tokens": 50,
            "temperature": 0.7,
            "timeout": 30
        }
    ]
    
    for model_config in models_to_try:
        try:
            llm = HuggingFaceEndpoint(
                huggingfacehub_api_token=api_key,
                **model_config
            )
            
            # Test with a simple prompt
            test_result = llm.invoke("Test")
            if test_result and test_result.strip():
                logger.info(f"Successfully initialized model: {model_config.get('repo_id', 'unknown')}")
                return llm
            
        except Exception as e:
            logger.warning(f"Failed to initialize {model_config.get('repo_id', 'unknown')}: {str(e)}")
            continue
    
    # Use enhanced fallback if all API models fail
    logger.warning("Using enhanced fallback LLM due to API failures")
    return EnhancedFallbackLLM()


def generate_fallback_response(user_input: str, context: str) -> str:
    """
    Generate a contextually appropriate fallback response.
    
    Args:
        user_input: The user's original question/message
        context: Available context from similar documents
        
    Returns:
        A contextually appropriate response
    """
    user_input_lower = user_input.lower()
    
    # Response mappings for different topics
    topic_responses = {
        'anxiety_attack': {
            'keywords': ['anxiety', 'attack', 'panic'],
            'response': """When experiencing anxiety attacks, try these techniques:
1. Practice deep breathing - breathe in slowly for 4 counts, hold for 4, exhale for 6
2. Use grounding techniques - name 5 things you can see, 4 you can hear, 3 you can touch
3. Remind yourself that panic attacks are temporary and will pass
4. Find a quiet, safe space if possible
5. Consider speaking with a mental health professional for personalized coping strategies

If you experience frequent or severe anxiety attacks, please consult with a healthcare provider."""
        },
        'anxiety': {
            'keywords': ['anxiety', 'anxious', 'panic', 'worry', 'worried'],
            'response': """I understand you're dealing with anxiety. Here are some general strategies that may help:
- Practice mindfulness and deep breathing exercises
- Maintain a regular sleep schedule and exercise routine
- Limit caffeine and alcohol intake
- Consider talking to a mental health professional
- Try relaxation techniques like progressive muscle relaxation

Remember, professional support can provide you with personalized strategies for managing anxiety."""
        },
        'depression': {
            'keywords': ['depression', 'depressed', 'sad', 'down', 'hopeless'],
            'response': """I understand you're experiencing difficult feelings. Depression is a common but serious condition. Here are some supportive steps:

- Reach out to trusted friends, family, or a mental health professional
- Try to maintain daily routines and engage in activities you usually enjoy
- Consider gentle exercise, even just a short walk
- Practice self-compassion and avoid self-criticism
- If you're having thoughts of self-harm, please contact a crisis helpline immediately

Professional help from a therapist or counselor can provide you with effective treatment options."""
        },
        'stress': {
            'keywords': ['stress', 'stressed', 'overwhelmed', 'pressure'],
            'response': """Feeling stressed or overwhelmed is common. Here are some strategies that might help:

- Break large tasks into smaller, manageable steps
- Practice time management and prioritization
- Take regular breaks and practice relaxation techniques
- Engage in physical activity to help reduce stress
- Talk to someone you trust about what you're experiencing

If stress is significantly impacting your daily life, a mental health professional can help you develop personalized coping strategies."""
        },
        'sleep': {
            'keywords': ['sleep', 'insomnia', 'tired', 'exhausted'],
            'response': """Sleep issues can significantly impact mental health. Here are some tips for better sleep:

- Maintain a consistent sleep schedule
- Create a relaxing bedtime routine
- Limit screen time before bed
- Keep your bedroom cool, dark, and quiet
- Avoid caffeine and large meals close to bedtime

If sleep problems persist, consult with a healthcare provider."""
        }
    }
    
    # Check for anxiety attack first (more specific)
    if 'attack' in user_input_lower and any(word in user_input_lower for word in ['anxiety', 'panic']):
        return topic_responses['anxiety_attack']['response']
    
    # Check other topics
    for topic, data in topic_responses.items():
        if topic != 'anxiety_attack' and any(word in user_input_lower for word in data['keywords']):
            return data['response']
    
    # Use context if available
    if context and "No specific context available" not in context:
        return """Based on the information available, I want to provide you with supportive guidance. While I can offer general mental health information, everyone's situation is unique.

I encourage you to speak with a qualified mental health professional who can provide personalized support and treatment options tailored to your specific needs.

Please know that seeking help is a sign of strength, and there are people and resources available to support you."""
    
    # General fallback
    return """I'm here to provide mental health support and information. I encourage you to reach out to a qualified mental health professional who can provide personalized support.

If you're in crisis or having thoughts of self-harm, please contact a crisis helpline or emergency services immediately. You don't have to face this alone."""


def format_response(similar_docs: List, user_input: str, chain: RunnableSequence) -> Generator:
    """
    Generate a streaming response using the LLM chain and similar documents.
    
    Args:
        similar_docs: List of similar documents from vector store
        user_input: User's question or message
        chain: The LLM chain object
        
    Yields:
        Server-sent event formatted responses
    """
    try:
        # Validate chain
        if chain is None:
            raise ConfigurationError("LLM chain not initialized")
        
        # Extract context from documents
        context = _extract_context(similar_docs)
        
        # Start streaming
        yield "data: " + json.dumps({"status": "start"}) + "\n\n"
        
        # Try direct LLM invocation if chain fails
        response = None
        try:
            # First try the chain
            response = _generate_with_retry(chain, context, user_input)
        except Exception as chain_error:
            logger.warning(f"Chain invocation failed completely: {chain_error}")
            
            # Try direct LLM invocation as last resort
            try:
                # Extract LLM from chain and invoke directly
                if hasattr(chain, 'middle') and len(chain.middle) > 0:
                    llm = chain.middle[0]
                elif hasattr(chain, 'steps') and len(chain.steps) > 1:
                    llm = chain.steps[1]
                else:
                    llm = None
                
                if llm:
                    prompt = f"Context: {context}\n\nQuestion: {user_input}\n\nAnswer:"
                    direct_response = llm.invoke(prompt)
                    
                    if isinstance(direct_response, str):
                        response = direct_response
                    elif hasattr(direct_response, 'content'):
                        response = direct_response.content
                    else:
                        response = str(direct_response)
                        
                    logger.info("Direct LLM invocation successful")
            except Exception as direct_error:
                logger.error(f"Direct LLM invocation also failed: {direct_error}")
        
        # Use fallback if all methods failed
        if not response or not response.strip():
            logger.info("Using fallback response generator")
            response = generate_fallback_response(user_input, context)
        
        # Stream the response
        yield from _stream_response(response)
        
        # Complete the stream
        yield "data: " + json.dumps({"status": "complete"}) + "\n\n"
        
    except Exception as e:
        logger.error(f"Error in format_response: {str(e)}")
        # Generate a helpful fallback even on error
        try:
            fallback = generate_fallback_response(user_input, "")
            yield from _stream_response(fallback)
            yield "data: " + json.dumps({"status": "complete"}) + "\n\n"
        except:
            error_msg = "I apologize, but I'm having trouble processing your request. Please try again."
            yield "data: " + json.dumps({"status": "error", "message": error_msg}) + "\n\n"


def _extract_context(similar_docs: List) -> str:
    """Extract context from similar documents."""
    if not similar_docs:
        return "No specific context available. Provide general mental health support."
    
    context_parts = []
    for doc in similar_docs[:3]:  # Limit to top 3 documents
        if hasattr(doc, 'page_content'):
            context_parts.append(doc.page_content)
    
    return "\n".join(context_parts) if context_parts else "No specific context available."


def _generate_with_retry(chain: RunnableSequence, context: str, user_input: str, max_retries: int = 3) -> Optional[str]:
    """
    Generate response with retry logic.
    
    Args:
        chain: The LLM chain
        context: Context string
        user_input: User input
        max_retries: Maximum number of retry attempts
        
    Returns:
        Generated response or None if all attempts fail
    """
    for attempt in range(max_retries):
        try:
            # Method 1: Try standard invoke
            try:
                result = chain.invoke({
                    "context": context,
                    "input": user_input
                })
            except StopIteration:
                # Method 2: Try with explicit iteration handling
                logger.info("StopIteration caught, trying alternative invocation")
                try:
                    # Get the components of the chain
                    prompt_result = chain.first.format(context=context, input=user_input)
                    llm = chain.middle[0] if hasattr(chain, 'middle') else chain.steps[1]
                    result = llm.invoke(prompt_result)
                except:
                    # Method 3: Try batch invocation with single input
                    logger.info("Trying batch invocation method")
                    results = chain.batch([{
                        "context": context,
                        "input": user_input
                    }])
                    result = results[0] if results else None
            
            # Handle empty generator or iterator
            if hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
                try:
                    result = ''.join(str(item) for item in result)
                except StopIteration:
                    logger.warning("Empty generator/iterator returned")
                    result = None
            
            # Extract and validate response
            if isinstance(result, dict):
                response = result.get("text", result.get("output", str(result)))
            else:
                response = str(result) if result else ""
            
            # Clean up response
            response = response.strip()
            
            # Remove any potential prompt artifacts
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            if "Context:" in response:
                response = response.split("Question:")[-1].split("Answer:")[-1].strip()
            
            if response:
                return response
            
        except StopIteration as e:
            logger.warning(f"StopIteration on attempt {attempt + 1}: Chain returned empty result")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed with {type(e).__name__}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    return None


def _stream_response(response: str) -> Generator:
    """Stream response as server-sent events."""
    words = response.split()
    for word in words:
        chunk = json.dumps({"status": "streaming", "token": word + " "})
        yield f"data: {chunk}\n\n"
        time.sleep(0.03)  # Simulate typing effect


def validate_input(user_input: str) -> Tuple[bool, str]:
    """
    Validate and sanitize user input.
    
    Args:
        user_input: User's input message
        
    Returns:
        Tuple of (is_valid, sanitized_input)
        
    Raises:
        ValidationError: If input validation fails
    """
    if not user_input:
        raise ValidationError("Input cannot be empty")
    
    if not isinstance(user_input, str):
        raise ValidationError(
            "Invalid input type",
            details={"expected": "string", "received": type(user_input).__name__}
        )
    
    # Sanitize input
    sanitized = user_input.strip()
    
    # Validate length
    if len(sanitized) < 2:
        raise ValidationError(
            "Input too short",
            details={"min_length": 2, "received_length": len(sanitized)}
        )
    
    if len(sanitized) > 300:
        raise ValidationError(
            "Input too long", 
            details={"max_length": 300, "received_length": len(sanitized)}
        )
    
    return True, sanitized