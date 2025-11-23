import os
import sys
import json
from dotenv import load_dotenv

from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.exception.custom_exception import DocumentPortalException 

# from logger import GLOBAL_LOGGER as log
from multi_doc_chat.logger.custom_logger import CustomLogger
log = CustomLogger().get_logger()
# from multi_doc_chat.logger.custom_logger import CustomLogger


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq


class ApiKeyManager:
    
    REQUIRED_KEYS = ["GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]
    
    def __init__(self):
        self.api_keys = {}
        # raw = os.getenv("apikeyliveclass")

        # if raw:
        #     try:
        #         parsed = json.loads(raw)
        #         if not isinstance(parsed, dict):
        #             raise ValueError("API_KEYS is not a valid JSON object")
        #         self.api_keys = parsed
        #         log.info("Loaded API_KEYS from ECS secret")
        #     except Exception as e:
        #         log.warning("Failed to parse API_KEYS as JSON", error=str(e))
        
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"loaded {key} from individual env var")
        
        # Final check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)
        
        log.info("API KEYS loaded", keys={k:v[:4] + "..." for k,v in self.api_keys.items()})
    
    def get(self, key:str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val

class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """
    
    def __init__(self) -> None:
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in Local mode : .env loaded")
        else:
            log.info("Running in Production mode")
        
        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys = list(self.config.keys()))
        
    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI.
        """
        try : 
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model = model_name)
            return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key = self.api_key_mgr.get("GOOGLE_API_KEY"))
        except Exception as e:
            log.error("Error loading embedding model", error = str(e))
            raise DocumentPortalException("failed to load the embedding model", sys)
    
    def load_llm(self):
        """
        Load and return the configured LLM model.
        """
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "google")
        
        if provide_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider {provider_key} not found in config")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name= llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)
        
        log.info("Loading LLM", provider=provider, model= model_name)
        
        if provider == "google":
            return ChatGoogleGenerativeAI(model=model_name, 
                                          google_api_key = self.api_key_mgr.get("GOOGLE_API_KEY"),
                                          temperature = temperature,
                                          max_output_tokens = max_tokens)
        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"), #type: ignore
                temperature=temperature,
            )
        # elif provider == "openai":
        #     return ChatOpenAI(
        #         model=model_name,
        #         api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #     )
        
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider : {provider}")
        
if __name__ == "__main__":
    loader = ModelLoader()
    
    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded : {embeddings}")
    result = embeddings.embed_query("Hello, how are you? What is your name?")
    print(f"Embedding result: {result}\n\n")
    print(f"size of embedding : {len(result)}")
        
